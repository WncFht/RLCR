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

    _tag_names = ["trl", "grpo"]

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
                    reward_func.config.pad_token_id = (
                        reward_processing_class.pad_token_id
                    )
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        self.group_confidence_by_accuracy = getattr(
            args, "group_confidence_by_accuracy", True
        )
        # 将奖励函数拆成“答案类/置信度类”，方便后续分别归一化
        confidence_keywords = ["brier", "confidence", "log_likelihood"]
        format_keywords = ["format"]
        self.confidence_reward_indices = []
        self.format_reward_indices = []
        for i, name in enumerate(self.reward_func_names):
            lname = name.lower()
            if any(keyword in lname for keyword in confidence_keywords):
                self.confidence_reward_indices.append(i)
            elif any(keyword in lname for keyword in format_keywords):
                self.format_reward_indices.append(i)
        self.answer_reward_indices = [
            i
            for i in range(len(self.reward_func_names))
            if i not in self.confidence_reward_indices
            and i not in self.format_reward_indices
        ]
        self.accuracy_reward_index = next(
            (
                i
                for i, name in enumerate(self.reward_func_names)
                if "accuracy" in name.lower()
            ),
            None,
        )
        if (
            self.group_confidence_by_accuracy
            and self.confidence_reward_indices
            and self.accuracy_reward_index is None
        ):
            raise ValueError(
                "Confidence rewards require 'accuracy_reward' to identify correct samples."
            )

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

        self._init_tag_token_ids()

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

    def _init_tag_token_ids(self):
        """预先把常用标签编码成 token 序列，后面解析 span 时无需重复分词"""
        tags = [
            "<think>",
            "</think>",
            "<answer>",
            "</answer>",
            "<analysis>",
            "</analysis>",
            "<confidence>",
            "</confidence>",
        ]
        self._tag_token_ids = {}
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            return
        for tag in tags:
            tokenized = tokenizer(tag, add_special_tokens=False)
            self._tag_token_ids[tag] = tokenized["input_ids"]
            print(
                f"Tag '{tag}' token IDs: {self._tag_token_ids[tag]}, length: {len(self._tag_token_ids[tag])}"
            )

    def _find_subsequence(self, sequence, pattern, start_idx=0):
        if not pattern or start_idx >= len(sequence):
            return -1
        pattern_len = len(pattern)
        limit = len(sequence) - pattern_len + 1
        for idx in range(start_idx, limit):
            if sequence[idx : idx + pattern_len] == pattern:
                return idx
        return -1

    def _find_last_subsequence(self, sequence, pattern, end_idx=None):
        if not pattern:
            return -1
        if end_idx is None or end_idx > len(sequence) - len(pattern):
            end_idx = len(sequence) - len(pattern)
        for idx in range(end_idx, -1, -1):
            if sequence[idx : idx + len(pattern)] == pattern:
                return idx
        return -1

    def _extract_token_spans(
        self, tokens: list[int]
    ) -> dict[str, Optional[tuple[int, int]]]:
        """
        仅依赖最后一个 <answer>...</answer> 来切分：
        - 之前 (含 think+answer) 归入答案部分；
        - 之后 (analysis+confidence) 归入置信度部分。
        """
        spans = {"answer": None, "confidence": None}
        if len(tokens) == 0 or self.processing_class is None:
            return spans

        decoded = self.processing_class.decode(tokens, skip_special_tokens=False)
        lowered = decoded.lower()
        close_str = "</answer>"
        open_str = "<answer>"
        answer_close_char = lowered.rfind(close_str)
        if answer_close_char == -1:
            return spans
        answer_open_char = lowered.rfind(open_str, 0, answer_close_char)
        if answer_open_char == -1:
            return spans

        # 将字符位置映射回 token 索引：重新编码前缀即可得到 token 数
        prefix_text = decoded[: answer_close_char + len(close_str)]
        prefix_tokens = self.processing_class(prefix_text, add_special_tokens=False)[
            "input_ids"
        ]
        ans_end = min(len(tokens), len(prefix_tokens))
        spans["answer"] = (0, ans_end)
        if ans_end < len(tokens):
            spans["confidence"] = (ans_end, len(tokens))
        return spans

    def _normalize_group_rewards(self, reward_tensor: torch.Tensor) -> torch.Tensor:
        if reward_tensor.numel() == 0:
            return reward_tensor
        grouped = reward_tensor.view(-1, self.num_generations)
        mean = grouped.mean(dim=1)
        std = grouped.std(dim=1)
        mean = mean.repeat_interleave(self.num_generations, dim=0)
        std = std.repeat_interleave(self.num_generations, dim=0)
        advantages = reward_tensor - mean
        if self.scale_rewards:
            advantages = advantages / (std + 1e-4)
        return advantages

    def _compute_confidence_advantages(
        self, confidence_rewards: torch.Tensor, accuracy_flags: torch.Tensor
    ) -> torch.Tensor:
        if confidence_rewards.numel() == 0:
            return confidence_rewards
        grouped_conf = confidence_rewards.view(-1, self.num_generations)
        if not self.group_confidence_by_accuracy:
            mean = grouped_conf.mean(dim=1, keepdim=True)
            std = grouped_conf.std(dim=1, unbiased=False, keepdim=True)
            adv = grouped_conf - mean
            if self.scale_rewards:
                adv = adv / (std + 1e-4)
            return adv.view(-1)
        grouped_acc = accuracy_flags.view(-1, self.num_generations)
        adv = torch.zeros_like(grouped_conf)
        for i in range(grouped_conf.size(0)):
            conf_row = grouped_conf[i]
            acc_row = grouped_acc[i]
            correct_mask = acc_row >= 0.5
            wrong_mask = ~correct_mask.bool()
            for mask in (correct_mask, wrong_mask):
                if mask.sum() == 0:
                    continue
                vals = conf_row[mask]
                mean = vals.mean()
                std = vals.std(unbiased=False)
                adv_vals = vals - mean
                if self.scale_rewards:
                    adv_vals = adv_vals / (std + 1e-4)
                adv[i, mask] = adv_vals
        return adv.view(-1)

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

        torch.cuda.empty_cache()

        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            if self.state.global_step >= -1:
                self.llm.wake_up()
                self.vllm_sleeping = False
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        sampling_params = SamplingParams(
            n=1,  # vLLM on each GPU generates only 1 in colocate mode
            temperature=self.temperature,
            max_tokens=self.max_completion_length,
        )

        with profiling_context(self, "vLLM.generate"):
            all_outputs = self.llm.generate(
                prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            # put to sleep
            if mode == "train":
                self.llm.sleep(level=1)
                self.vllm_sleeping = True
                self.accelerator.wait_for_everyone()

        completion_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        # Pad the completions, and concatenate them with the prompts
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id
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

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # 为每条 completion 准备 think+answer 与 analysis+confidence 的 token mask
        answer_token_mask = torch.zeros_like(completion_mask, dtype=torch.float32)
        confidence_token_mask = torch.zeros_like(completion_mask, dtype=torch.float32)
        valid_answer_split = torch.ones(
            completion_mask.size(0), dtype=torch.bool, device=completion_mask.device
        )
        split_positions = torch.full(
            (completion_mask.size(0),), float("nan"), device=completion_mask.device
        )
        split_ratios = torch.full_like(split_positions, float("nan"))
        for idx, tokens in enumerate(completion_ids_list):
            spans = self._extract_token_spans(tokens)
            seq_len = len(tokens)
            if spans.get("answer") is None:
                valid_answer_split[idx] = False
                continue
            start, end = spans["answer"]
            start = max(0, min(start, seq_len))
            end = max(start, min(end, seq_len))
            answer_token_mask[idx, start:end] = 1.0
            if end > 0:
                split_positions[idx] = end - 1
                if seq_len > 0:
                    split_ratios[idx] = end / max(seq_len, 1)
            if spans.get("confidence") is not None:
                c_start, c_end = spans["confidence"]
                c_start = max(0, min(c_start, seq_len))
                c_end = max(c_start, min(c_end, seq_len))
                confidence_token_mask[idx, c_start:c_end] = 1.0

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        # span mask 也需遵从 completion_mask，屏蔽掉 truncation/padding 的 token
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
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
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
                    # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the number
                    # of generations
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
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        valid_answer_split = gather(valid_answer_split)

        weight_vector = self.reward_weights.to(rewards_per_func.device)

        def _combine_rewards(indices: list[int], mask: torch.Tensor) -> torch.Tensor:
            if not indices:
                return torch.zeros_like(mask, dtype=rewards_per_func.dtype)
            weights = weight_vector[indices].unsqueeze(0)
            selected = rewards_per_func[:, indices]
            return (selected * weights).sum(dim=1) * mask.float()

        mask_float = valid_answer_split.to(
            rewards_per_func.device, dtype=rewards_per_func.dtype
        )
        rewards_per_func = rewards_per_func * mask_float.unsqueeze(1)
        valid_answer_ratio = mask_float.mean().item() if mask_float.numel() > 0 else 0.0
        self._metrics[mode]["valid_answer_ratio"].append(valid_answer_ratio)

        answer_rewards = _combine_rewards(self.answer_reward_indices, mask_float)
        confidence_rewards = _combine_rewards(
            self.confidence_reward_indices, mask_float
        )
        format_rewards = _combine_rewards(self.format_reward_indices, mask_float)
        total_rewards = answer_rewards + confidence_rewards + format_rewards

        accuracy_scores = (
            rewards_per_func[:, self.accuracy_reward_index]
            if self.accuracy_reward_index is not None
            else torch.zeros_like(mask_float)
        )

        if self.group_confidence_by_accuracy and self.accuracy_reward_index is not None:
            grouped_acc = accuracy_scores.view(-1, self.num_generations)
            grouped_valid = mask_float.view(-1, self.num_generations).bool()
            valid_counts = grouped_valid.sum(dim=1)
            has_valid = valid_counts > 0
            correct_counts = ((grouped_acc >= 0.5) & grouped_valid).sum(dim=1).float()
            wrong_counts = ((grouped_acc < 0.5) & grouped_valid).sum(dim=1).float()
            correct_ratio = torch.full(
                (grouped_acc.size(0),), float("nan"), device=mask_float.device
            )
            wrong_ratio = torch.full_like(correct_ratio, float("nan"))
            if has_valid.any():
                denom = valid_counts[has_valid].float()
                correct_ratio[has_valid] = correct_counts[has_valid] / denom.clamp(
                    min=1.0
                )
                wrong_ratio[has_valid] = wrong_counts[has_valid] / denom.clamp(min=1.0)
                self._metrics[mode]["confidence/correct_group_ratio"].append(
                    torch.nanmean(correct_ratio).item()
                )
                self._metrics[mode]["confidence/wrong_group_ratio"].append(
                    torch.nanmean(wrong_ratio).item()
                )

        answer_advantages_full = (
            self._normalize_group_rewards(answer_rewards)
            if self.answer_reward_indices
            else torch.zeros_like(mask_float)
        )
        confidence_advantages_full = (
            self._compute_confidence_advantages(confidence_rewards, accuracy_scores)
            if self.confidence_reward_indices
            else torch.zeros_like(mask_float)
        )
        format_advantages_full = (
            self._normalize_group_rewards(format_rewards)
            if self.format_reward_indices
            else torch.zeros_like(mask_float)
        )

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        answer_advantages = answer_advantages_full[process_slice]
        confidence_advantages = confidence_advantages_full[process_slice]
        format_advantages = format_advantages_full[process_slice]
        # Token masks are already local tensors; slicing them with global offsets would
        # produce empty tensors on non-zero ranks. Keep the local shapes untouched so
        # they stay aligned with the current process's batch.
        answer_token_mask_local = answer_token_mask
        confidence_token_mask_local = confidence_token_mask
        format_token_mask_local = completion_mask.float()

        token_advantages = (
            answer_advantages.unsqueeze(1) * answer_token_mask_local
            + confidence_advantages.unsqueeze(1) * confidence_token_mask_local
            + format_advantages.unsqueeze(1) * format_token_mask_local
        )
        advantages = token_advantages

        # Log the metrics
        mean_grouped_rewards = total_rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = total_rewards.view(-1, self.num_generations).std(dim=1)
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
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

        agg_split_positions = self.accelerator.gather_for_metrics(split_positions)
        agg_split_ratios = self.accelerator.gather_for_metrics(split_ratios)
        valid_split_positions = agg_split_positions[torch.isfinite(agg_split_positions)]
        valid_split_ratios = agg_split_ratios[torch.isfinite(agg_split_ratios)]
        if valid_split_positions.numel() > 0:
            self._metrics[mode]["spans/split_position"].append(
                valid_split_positions.float().mean().item()
            )
        if valid_split_ratios.numel() > 0:
            self._metrics[mode]["spans/split_ratio"].append(
                valid_split_ratios.float().mean().item()
            )

        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(
            agg_completion_mask
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_mask.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_mask.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_mask.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            reward_values = rewards_per_func[:, i]
            mean_rewards = torch.nanmean(reward_values).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            batch_std_rewards = nanstd(reward_values).item()
            self._metrics[mode][f"rewards/{reward_func_name}/batch_std"].append(
                batch_std_rewards
            )

            group_std_rewards = 0.0
            group_size = int(self.num_generations)
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
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        if self.answer_reward_indices:
            answer_group_mean = answer_rewards.view(-1, self.num_generations).mean(
                dim=1
            )
            self._metrics[mode]["reward/answer"].append(answer_group_mean.mean().item())
        if self.confidence_reward_indices:
            confidence_group_mean = confidence_rewards.view(
                -1, self.num_generations
            ).mean(dim=1)
            self._metrics[mode]["reward/confidence"].append(
                confidence_group_mean.mean().item()
            )
        if self.format_reward_indices:
            format_group_mean = format_rewards.view(-1, self.num_generations).mean(
                dim=1
            )
            self._metrics[mode]["reward/format"].append(format_group_mean.mean().item())

        # Log prompt and completion texts
        num_completions_to_log = self.args.num_completions_to_log
        self._textual_logs["step"].extend(
            [str(self.state.global_step)] * num_completions_to_log
        )
        self._textual_logs["prompt"].extend(
            gather_object(prompts_text)[0:num_completions_to_log]
        )
        self._textual_logs["completion"].extend(
            gather_object(completions_text)[0:num_completions_to_log]
        )
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
            # pad in rare cases where advantages is shorter than the logits (shouldn't happen but be safe)
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

        if self.accelerator.is_main_process and self.log_completions:
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
