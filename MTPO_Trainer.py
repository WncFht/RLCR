# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import textwrap
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import gather, gather_object, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from trainer_utils import (
    RepeatSampler,
    disable_dropout_in_model,
    nanmax,
    nanmin,
    nanstd,
    profiling_context,
    profiling_decorator,
    shuffle_tensor_dict,
    split_tensor_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    prepare_deepspeed,
    selective_log_softmax,
)

from trl import (
    GRPOConfig,
    SyncRefModelCallback,
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)

if is_peft_available():
    pass

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class CustomTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "mtpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.config._name_or_path, padding_side="left"
            )
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(
                reward_funcs[i], nn.Module
            ):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(
                    reward_funcs[i].config._name_or_path.split("/")[-1]
                )
            else:
                try:
                    self.reward_func_names.append(reward_funcs[i].__name__)
                except:
                    self.reward_func_names.append(reward_funcs[i].func.__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(len(args.reward_weights))}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(
                len(reward_funcs), dtype=torch.float32
            ) / len(reward_funcs)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions
        self.vllm_sleeping = False

        self.log_completions = args.log_completions
        self.print_completion = getattr(args, "print_completion", False)

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        # Multi-step
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        self._step = 0
        self._buffered_inputs = None

        # MTPO specific
        self.num_answer_generations = getattr(args, "num_answer_generations", None)
        self.num_confidence_generations = getattr(
            args, "num_confidence_generations", None
        )
        if (
            self.num_answer_generations is None
            or self.num_confidence_generations is None
        ):
            raise ValueError(
                "MTPOConfig must provide num_answer_generations and num_confidence_generations."
            )
        if (
            self.num_answer_generations * self.num_confidence_generations
            != self.num_generations
        ):
            raise ValueError(
                "MTPO requires args.num_generations == num_answer_generations * num_confidence_generations "
                f"(got {self.num_generations} vs {self.num_answer_generations}*{self.num_confidence_generations})."
            )

        self.max_answer_length = getattr(args, "max_answer_length", None)
        self.max_confidence_length = getattr(args, "max_confidence_length", None)
        if self.max_answer_length is None or self.max_confidence_length is None:
            # Fallback: split total budget evenly.
            half = max(1, int(self.max_completion_length // 2))
            self.max_answer_length = half
            self.max_confidence_length = self.max_completion_length - half

        def _stop_str_variants(stop_str: str) -> list[str]:
            # Many tokenizers merge leading whitespace into the following token, so the model often emits
            # " </answer>" or "\n</answer>" as a single token. We include common variants for robust stopping
            # and robust token-level detection.
            candidates = [stop_str]
            if stop_str and not stop_str.startswith(" "):
                candidates.append(f" {stop_str}")
            candidates.append(f"\n{stop_str}")
            if stop_str and not stop_str.startswith(" "):
                candidates.append(f"\n {stop_str}")
            # de-duplicate while preserving order
            seen = set()
            out = []
            for s in candidates:
                if s not in seen:
                    out.append(s)
                    seen.add(s)
            return out

        self.answer_stop_str = getattr(args, "answer_stop_str", "</answer>")
        self.confidence_stop_str = getattr(args, "confidence_stop_str", "</confidence>")
        self._answer_stop_str_variants = _stop_str_variants(self.answer_stop_str)
        self._confidence_stop_str_variants = _stop_str_variants(
            self.confidence_stop_str
        )
        self.apply_answer_loss_on_first_confidence_only = getattr(
            args, "apply_answer_loss_on_first_confidence_only", True
        )
        if processing_class is not None:
            self._answer_stop_token_id_variants = [
                processing_class(s, add_special_tokens=False)["input_ids"]
                for s in self._answer_stop_str_variants
            ]
            self._confidence_stop_token_id_variants = [
                processing_class(s, add_special_tokens=False)["input_ids"]
                for s in self._confidence_stop_str_variants
            ]
            self._answer_stop_token_id_variants = [
                ids for ids in self._answer_stop_token_id_variants if ids
            ]
            self._confidence_stop_token_id_variants = [
                ids for ids in self._confidence_stop_token_id_variants if ids
            ]
            # Keep the original single-sequence attributes for backward compatibility / readability.
            self._answer_stop_token_ids = (
                self._answer_stop_token_id_variants[0]
                if self._answer_stop_token_id_variants
                else []
            )
            self._confidence_stop_token_ids = (
                self._confidence_stop_token_id_variants[0]
                if self._confidence_stop_token_id_variants
                else []
            )
        else:
            self._answer_stop_token_ids = []
            self._confidence_stop_token_ids = []
            self._answer_stop_token_id_variants = []
            self._confidence_stop_token_id_variants = []

        # Split reward functions into answer / confidence parts for MTPO baselines.
        confidence_keywords = ["brier", "confidence", "log_likelihood"]
        self.answer_reward_indices = []
        self.confidence_reward_indices = []
        for i, name in enumerate(self.reward_func_names):
            lname = str(name).lower()
            if "format_confidence" in lname:
                self.confidence_reward_indices.append(i)
            elif "format_answer" in lname:
                self.answer_reward_indices.append(i)
            elif any(k in lname for k in confidence_keywords):
                self.confidence_reward_indices.append(i)
            else:
                self.answer_reward_indices.append(i)

        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_id, **model_init_kwargs
            )
            self.ref_model = self.accelerator.prepare(self.ref_model)
            self.ref_model.eval()
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
            # send to accelerator device
            self.ref_model = self.ref_model.to(self.accelerator.device)
            self.ref_model.eval()

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.completion_logging_steps = args.completion_logging_steps
        self.wandb_counter = 0
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        self._textual_logs = {
            "step": deque(maxlen=50000),
            "prompt": deque(maxlen=50000),
            "completion": deque(maxlen=50000),
            "rewards": defaultdict(lambda: deque(maxlen=50000)),
        }

        set_seed(args.seed, device_specific=True)

        if not is_vllm_available():
            raise ImportError(
                "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                "`pip install vllm` to use it."
            )

        if self.vllm_mode == "colocate":
            self.llm = LLM(
                model=model.name_or_path,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_num_seqs=self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps,
                max_model_len=self.max_prompt_length + self.max_completion_length,
                distributed_executor_backend="external_launcher",
                # Feed identical seed for tp groups to ensure sampling results are the same across workers
                seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                enable_sleep_mode=True,
            )

            self._last_loaded_step = (
                -1
            )  # tag to avoid useless loading during grad accumulation
            self.model_name_or_path = model.name_or_path
            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()

        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if args.sync_ref_model:
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(
                        reward_func, self.accelerator
                    )
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size
            * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(
        self, model: PreTrainedModel, args: GRPOConfig
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )
        if use_reentrant:
            model.enable_input_require_grads()
        return model

    @profiling_decorator
    def _get_last_hidden_state(
        self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None
    ):
        last_hidden_state = unwrapped_model.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[
                :, -logits_to_keep:, :
            ]  # (B, logits_to_keep, H)
        return last_hidden_state

    @profiling_decorator
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        mode="train",
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(
            0
        )  # Chunk inputs into smaller batches to reduce memory peak
        batch_size = (
            self.args.per_device_train_batch_size
        )  ##Set small batch size to reduce memory peak
        if mode == "eval":
            batch_size = 1
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(
                logits, input_ids_batch
            )  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        gather_if_zero3 = nullcontext
        for name, param in self.model.named_parameters():
            with gather_if_zero3([param]):
                if self.vllm_mode == "server" and self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(name, param.data)
                elif self.vllm_mode == "colocate":
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights([(name, param.data)])

        if self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                generation_batch = shuffle_tensor_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation
                )
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]], eval=False
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Extract unpadded token ids for vLLM token prompts (avoid re-tokenization mismatch).
        prompt_token_ids = [
            prompt_ids[i][prompt_mask[i].bool()].tolist() for i in range(len(inputs))
        ]

        torch.cuda.empty_cache()

        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            if self.state.global_step >= -1:
                self.llm.wake_up()
                self.vllm_sleeping = False
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        def _truncate_to_last_subseq_any(
            tokens: list[int], subseqs: list[list[int]]
        ) -> tuple[list[int], bool]:
            best_end = None
            for subseq in subseqs:
                if not subseq:
                    continue
                # Search from the end (small lists, python loop is fine).
                max_start = len(tokens) - len(subseq)
                for start in range(max_start, -1, -1):
                    if tokens[start : start + len(subseq)] == subseq:
                        end = start + len(subseq)
                        if best_end is None or end > best_end:
                            best_end = end
                        break
            if best_end is None:
                return tokens, False
            return tokens[:best_end], True

        def _as_list(tokens: Any) -> list[int]:
            # vLLM may return tuples; normalize to a Python list for downstream concatenation/slicing.
            if isinstance(tokens, list):
                return tokens
            if isinstance(tokens, tuple):
                return list(tokens)
            if isinstance(tokens, torch.Tensor):
                return [int(x) for x in tokens.tolist()]
            # Fall back to best-effort conversion (e.g., numpy arrays).
            return [int(x) for x in list(tokens)]

        def _strip_eos(tokens: Any) -> list[int]:
            tokens = _as_list(tokens)
            eos = self.processing_class.eos_token_id
            if eos in tokens:
                return tokens[: tokens.index(eos)]
            return tokens

        def _truncate_to_last_closed_tag(
            tokens: list[int], open_tag: str, close_tag: str
        ) -> tuple[list[int], bool]:
            # Mirror SMCR logic: find the last <tag>...</tag> boundary in decoded text and
            # map back to token indices by re-tokenizing the prefix substring.
            if not tokens or self.processing_class is None:
                return tokens, False
            decoded = self.processing_class.decode(tokens, skip_special_tokens=False)
            lowered = decoded.lower()
            close_idx = lowered.rfind(close_tag)
            if close_idx == -1:
                return tokens, False
            open_idx = lowered.rfind(open_tag, 0, close_idx)
            if open_idx == -1:
                return tokens, False
            prefix_text = decoded[: close_idx + len(close_tag)]
            prefix_tokens = self.processing_class(
                prefix_text, add_special_tokens=False
            )["input_ids"]
            end = min(len(tokens), len(prefix_tokens))
            return tokens[:end], True

        # -------------------------
        # Round 1: think + answer
        # -------------------------
        # Inputs are arranged as [answer0_conf0..confH-1, answer1_conf0.., ...],
        # so we only need 1 prompt per answer slot.
        H = int(self.num_confidence_generations)
        if len(inputs) % H != 0:
            raise ValueError(
                f"Batch size ({len(inputs)}) must be divisible by num_confidence_generations ({H})."
            )
        answer_slot_indices = list(range(0, len(inputs), H))
        answer_slot_prompts: list[TokensPrompt] = [
            {"prompt_token_ids": prompt_token_ids[i]} for i in answer_slot_indices
        ]
        sampling_params_answer = SamplingParams(
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_answer_length,
            stop=self._answer_stop_str_variants,
            include_stop_str_in_output=True,
        )

        with profiling_context(self, "vLLM.generate_answer"):
            answer_outputs = self.llm.generate(
                answer_slot_prompts,
                sampling_params=sampling_params_answer,
                use_tqdm=False,
            )

        answer_completion_ids_per_slot: list[list[int]] = []
        valid_answer_slots: list[bool] = []
        answer_close_found_flags: list[bool] = []
        answer_finish_stop_flags: list[bool] = []
        for outputs in answer_outputs:
            for output in outputs.outputs:
                tok = _strip_eos(output.token_ids)
                tok, found_close = _truncate_to_last_closed_tag(
                    tok, "<answer>", "</answer>"
                )
                finish_reason = getattr(output, "finish_reason", None)
                answer_close_found_flags.append(bool(found_close))
                answer_finish_stop_flags.append(finish_reason == "stop")
                if (
                    (not found_close)
                    and finish_reason == "stop"
                    and self.processing_class.eos_token_id is not None
                    and (len(tok) == 0 or tok[-1] != self.processing_class.eos_token_id)
                ):
                    # Do not inject tag tokens; just terminate the segment.
                    tok = tok + [self.processing_class.eos_token_id]
                answer_completion_ids_per_slot.append(tok)
                valid_answer_slots.append(bool(found_close))

        if answer_close_found_flags:
            self._metrics[mode]["mtpo/answer_stop_token_found_ratio"].append(
                float(sum(answer_close_found_flags)) / len(answer_close_found_flags)
            )
        if answer_finish_stop_flags:
            self._metrics[mode]["mtpo/answer_finish_reason_stop_ratio"].append(
                float(sum(answer_finish_stop_flags)) / len(answer_finish_stop_flags)
            )

        # -------------------------
        # Round 2: analysis + confidence (conditioned on round-1 answer)
        # -------------------------
        confidence_prompts: list[TokensPrompt] = []
        slot_id_for_sample = []
        for i in range(len(inputs)):
            slot_id = i // H
            slot_id_for_sample.append(slot_id)
            prefix = prompt_token_ids[i] + answer_completion_ids_per_slot[slot_id]
            confidence_prompts.append({"prompt_token_ids": prefix})

        sampling_params_conf = SamplingParams(
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_confidence_length,
            stop=self._confidence_stop_str_variants,
            include_stop_str_in_output=True,
        )
        with profiling_context(self, "vLLM.generate_confidence"):
            confidence_outputs = self.llm.generate(
                confidence_prompts, sampling_params=sampling_params_conf, use_tqdm=False
            )

        if mode == "train":
            self.llm.sleep(level=1)
            self.vllm_sleeping = True
            self.accelerator.wait_for_everyone()

        confidence_completion_ids: list[list[int]] = []
        for outputs in confidence_outputs:
            for output in outputs.outputs:
                tok = _strip_eos(output.token_ids)
                tok, found_stop = _truncate_to_last_closed_tag(
                    tok, "<confidence>", "</confidence>"
                )
                finish_reason = getattr(output, "finish_reason", None)
                has_text_stop = False
                if (
                    (not found_stop)
                    and finish_reason == "stop"
                    and self.confidence_stop_str
                ):
                    decoded = self.processing_class.decode(
                        tok, skip_special_tokens=True
                    )
                    has_text_stop = decoded.rstrip().endswith(self.confidence_stop_str)
                if (not found_stop) and (not has_text_stop) and finish_reason == "stop":
                    # Do not inject tag tokens; just mark completion as ended.
                    pass
                if self.processing_class.eos_token_id is not None and (
                    len(tok) == 0 or tok[-1] != self.processing_class.eos_token_id
                ):
                    tok = tok + [self.processing_class.eos_token_id]
                confidence_completion_ids.append(tok)

        # -------------------------
        # Combine two rounds into one completion sequence.
        # -------------------------
        combined_completion_tensors: list[torch.Tensor] = []
        stage1_lens: list[int] = []
        stage2_lens: list[int] = []
        valid_mask_local = torch.zeros(len(inputs), dtype=torch.float32, device=device)
        for i in range(len(inputs)):
            slot_id = slot_id_for_sample[i]
            stage1 = answer_completion_ids_per_slot[slot_id]
            stage2 = confidence_completion_ids[i]
            combined = stage1 + stage2
            combined_completion_tensors.append(torch.tensor(combined, device=device))
            stage1_lens.append(len(stage1))
            stage2_lens.append(len(stage2))
            valid_mask_local[i] = 1.0 if valid_answer_slots[slot_id] else 0.0

        completion_ids = pad(
            combined_completion_tensors,
            padding_value=self.processing_class.pad_token_id,
        )
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Token-level masks for round-1 / round-2 losses
        answer_token_mask = torch.zeros_like(completion_mask, dtype=torch.float32)
        confidence_token_mask = torch.zeros_like(completion_mask, dtype=torch.float32)
        for i in range(len(inputs)):
            a_len = stage1_lens[i]
            c_len = stage2_lens[i]
            if a_len > 0 and (
                (not self.apply_answer_loss_on_first_confidence_only) or (i % H == 0)
            ):
                answer_token_mask[i, :a_len] = 1.0
            if c_len > 0:
                confidence_token_mask[i, a_len : a_len + c_len] = 1.0

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        answer_token_mask = answer_token_mask * completion_mask
        confidence_token_mask = confidence_token_mask * completion_mask

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if (
                self.num_iterations > 1
                or self.args.steps_per_generation
                > self.args.gradient_accumulation_steps
            ):
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    mode=mode,
                )
            else:
                old_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            fixed_prompts = []
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                if prompt and prompt[-1]["role"] == "assistant":
                    bootstrap = prompt[-1]["content"]
                    fixed_prompts.append(prompt[:-1])
                else:
                    bootstrap = ""
                    fixed_prompts.append(prompt)
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
            prompts = fixed_prompts
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
            )
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c} for p, c in zip(prompts, completions)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                            :, 0
                        ]  # Shape (B*G,)
                else:
                    keys = [
                        key
                        for key in inputs[0]
                        if key not in ["prompt", "completion", "completion_ids"]
                    ]
                    reward_kwargs = {
                        key: [example[key] for example in inputs] for key in keys
                    }
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        completion_ids=completion_ids_list,
                        **reward_kwargs,
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # Gather the reward per function: crucial for baselines across processes
        rewards_per_func = gather(rewards_per_func)
        rewards_per_func = torch.nan_to_num(rewards_per_func, nan=0.0)
        valid_mask = gather(valid_mask_local)

        # Apply validity mask (invalid stage-1 answers contribute 0 reward)
        rewards_per_func = rewards_per_func * valid_mask.unsqueeze(1)
        valid_ratio = valid_mask.mean().item() if valid_mask.numel() else 0.0
        self._metrics[mode]["valid_answer_ratio"].append(valid_ratio)

        weight_vector = self.reward_weights.to(rewards_per_func.device)

        def _combine_rewards(indices: list[int]) -> torch.Tensor:
            if not indices:
                return torch.zeros(
                    rewards_per_func.size(0),
                    dtype=rewards_per_func.dtype,
                    device=rewards_per_func.device,
                )
            weights = weight_vector[indices].unsqueeze(0)
            selected = rewards_per_func[:, indices]
            return (selected * weights).sum(dim=1)

        answer_rewards = _combine_rewards(self.answer_reward_indices)
        confidence_rewards = _combine_rewards(self.confidence_reward_indices)
        total_rewards = answer_rewards + confidence_rewards

        G = int(self.num_answer_generations)
        H = int(self.num_confidence_generations)
        if total_rewards.numel() % (G * H) != 0:
            raise ValueError(
                f"Total rewards length ({total_rewards.numel()}) must be divisible by G*H ({G * H})."
            )

        # Answer baseline: group by prompt, across G answers (collapse H by mean)
        grouped_answer = answer_rewards.view(-1, G, H)
        answer_per_answer = grouped_answer.mean(dim=2)  # (P, G)
        ans_mean = answer_per_answer.mean(dim=1, keepdim=True)
        ans_std = answer_per_answer.std(dim=1, keepdim=True, unbiased=False)
        answer_adv_per_answer = answer_per_answer - ans_mean
        if self.scale_rewards:
            answer_adv_per_answer = answer_adv_per_answer / (ans_std + 1e-4)
        answer_adv = answer_adv_per_answer.unsqueeze(2).expand(-1, -1, H)
        if self.apply_answer_loss_on_first_confidence_only:
            apply_mask = torch.zeros_like(answer_adv)
            apply_mask[:, :, 0] = 1.0
            answer_adv = answer_adv * apply_mask

        # Confidence baseline: group by answer, across H confidence samples
        grouped_conf = confidence_rewards.view(-1, G, H)
        conf_mean = grouped_conf.mean(dim=2, keepdim=True)
        conf_std = grouped_conf.std(dim=2, keepdim=True, unbiased=False)
        confidence_adv = grouped_conf - conf_mean
        if self.scale_rewards:
            confidence_adv = confidence_adv / (conf_std + 1e-4)

        answer_adv_flat = answer_adv.reshape(-1)
        confidence_adv_flat = confidence_adv.reshape(-1)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        answer_adv_local = answer_adv_flat[process_slice]
        confidence_adv_local = confidence_adv_flat[process_slice]

        advantages = (
            answer_adv_local.unsqueeze(1) * answer_token_mask
            + confidence_adv_local.unsqueeze(1) * confidence_token_mask
        )

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_mask = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        )
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_mask.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_mask.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_mask.float().max().item()
        )

        # Calculate mean reward per function (global)
        for i, reward_func_name in enumerate(self.reward_func_names):
            reward_values = rewards_per_func[:, i]
            self._metrics[mode][f"rewards/{reward_func_name}"].append(
                torch.nanmean(reward_values).item()
            )
            batch_std_rewards = nanstd(reward_values).item()
            self._metrics[mode][f"rewards/{reward_func_name}/batch_std"].append(
                batch_std_rewards
            )

            group_std_rewards = 0.0
            group_size = int(G * H)
            if group_size > 0:
                num_groups = reward_values.numel() // group_size
                if num_groups > 0:
                    grouped = reward_values[: num_groups * group_size].view(
                        num_groups, group_size
                    )
                    valid = ~torch.isnan(grouped)
                    counts = valid.sum(dim=1)
                    has_enough = counts > 1
                    if has_enough.any():
                        mean = torch.nanmean(grouped, dim=1, keepdim=True)
                        var = torch.nanmean((grouped - mean) ** 2, dim=1)
                        counts_f = counts.to(var.dtype)
                        denom = (counts_f - 1).clamp(min=1)
                        var = var * counts_f / denom
                        group_std_rewards = torch.sqrt(var)[has_enough].mean().item()
            self._metrics[mode][f"rewards/{reward_func_name}/group_std"].append(
                group_std_rewards
            )

            answer_std_rewards = 0.0
            group_size = int(G * H)
            if group_size > 0:
                num_groups = reward_values.numel() // group_size
                if num_groups > 0:
                    grouped = reward_values[: num_groups * group_size].view(
                        num_groups, G, H
                    )
                    valid = ~torch.isnan(grouped)
                    counts = valid.sum(dim=2)
                    has_enough = counts > 1
                    if has_enough.any():
                        mean = torch.nanmean(grouped, dim=2, keepdim=True)
                        var = torch.nanmean((grouped - mean) ** 2, dim=2)
                        answer_std_rewards = torch.sqrt(var)[has_enough].mean().item()
            self._metrics[mode][f"rewards/{reward_func_name}/answer_std"].append(
                answer_std_rewards
            )

        # Aggregate total reward stats (per prompt group)
        grouped_total = total_rewards.view(-1, G * H)
        self._metrics[mode]["reward"].append(grouped_total.mean(dim=1).mean().item())
        self._metrics[mode]["reward_std"].append(grouped_total.std(dim=1).mean().item())

        # Log prompt and completion texts
        if self.log_completions or self.print_completion:
            num_completions_to_log = self.args.num_completions_to_log
            gathered_prompts = gather_object(prompts_text)[0:num_completions_to_log]
            gathered_completions = gather_object(completions_text)[
                0:num_completions_to_log
            ]
            if self.accelerator.is_main_process:
                self._textual_logs["step"].extend(
                    [str(self.state.global_step)] * num_completions_to_log
                )
                self._textual_logs["prompt"].extend(gathered_prompts)
                self._textual_logs["completion"].extend(gathered_completions)
                for i, name in enumerate(self.reward_func_names):
                    self._textual_logs["rewards"][name].extend(
                        rewards_per_func[:, i].tolist()[0:num_completions_to_log]
                    )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Log the metrics
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            torch.cuda.empty_cache()
            if not self.vllm_sleeping:
                self.llm.sleep()
                self.accelerator.wait_for_everyone()
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )

        # sum completion masks to find max dimension
        max_dim = completion_mask.sum(1).max()
        # now clip everything to that dim
        completion_mask = completion_mask[:, :max_dim]
        completion_ids = completion_ids[:, :max_dim]
        advantages = inputs["advantages"]
        if advantages.size(1) > max_dim:
            advantages = advantages[:, :max_dim]
        elif advantages.size(1) < max_dim:
            pad_width = max_dim - advantages.size(1)
            pad_tensor = torch.zeros(
                advantages.size(0),
                pad_width,
                dtype=advantages.dtype,
                device=advantages.device,
            )
            advantages = torch.cat([advantages, pad_tensor], dim=1)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        with torch.no_grad():
            completion_lengths = completion_mask.sum(dim=1).clamp(min=1)
            entropy_per_sample = -(
                (per_token_logps * completion_mask).sum(dim=1) / completion_lengths
            )
            gathered_entropy = self.accelerator.gather_for_metrics(entropy_per_sample)
            self._metrics[mode]["policy/entropy_per_token"].append(
                gathered_entropy.nanmean().item()
            )
            self._metrics[mode]["policy/entropy_per_token_std"].append(
                nanstd(gathered_entropy).item()
            )

        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        old_per_token_logps = (
            per_token_logps.detach()
            if inputs["old_per_token_logps"] is None
            else inputs["old_per_token_logps"]
        )
        if (
            old_per_token_logps is not None
            and old_per_token_logps.size(1) != logits_to_keep
        ):
            old_per_token_logps = old_per_token_logps[:, :logits_to_keep]
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            # Use clamp instead of min to handle tensor-float comparison
            per_token_loss1 = torch.clamp(coef_1, max=self.args.delta) * advantages
        else:
            # Original GRPO clipping (only lower bound implicitly applied by the final min)
            per_token_loss1 = coef_1 * advantages

        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).nanmean().item()
            )

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        # Return placeholder zero loss without running compute_loss
        return torch.tensor(0.0, device=self.accelerator.device), None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.print_completion
            and self.completion_logging_steps is not None
            and self.state.global_step % self.completion_logging_steps == 0
            and len(self._textual_logs["step"]) > 0
        ):
            out_dir = os.path.join(self.args.output_dir, "completions")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir, f"completions_{self.state.global_step}.table.json"
            )

            steps = list(self._textual_logs["step"])
            prompts = list(self._textual_logs["prompt"])
            completions = list(self._textual_logs["completion"])
            reward_names = list(self.reward_func_names)
            rewards = {
                name: list(self._textual_logs["rewards"][name]) for name in reward_names
            }

            n = min(len(steps), len(prompts), len(completions))
            columns = ["step", "prompt", "completion", *reward_names]
            data = []
            for idx in range(n):
                row = [steps[idx], prompts[idx], completions[idx]]
                for name in reward_names:
                    row.append(rewards[name][idx] if idx < len(rewards[name]) else None)
                data.append(row)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"columns": columns, "data": data, "mode": mode},
                    f,
                    ensure_ascii=False,
                )

            self._textual_logs["step"].clear()
            self._textual_logs["prompt"].clear()
            self._textual_logs["completion"].clear()
            for name in list(self._textual_logs["rewards"].keys()):
                self._textual_logs["rewards"][name].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and (not self.print_completion)
        ):
            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                import pandas as pd

                table = {
                    "step": self._textual_logs["step"],
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(
            self.model.config._name_or_path
        ):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url()
            if is_wandb_available() and wandb.run is not None
            else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
