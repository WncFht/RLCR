from __future__ import annotations

from typing import Any, Callable, Optional, Union

import MTPO_Trainer as mtpo
import torch
from reward_fns import accuracy_reward

RewardFunc = Union[str, mtpo.PreTrainedModel, Callable[[list, list], list[float]]]


class CustomTrainer(mtpo.CustomTrainer):
    """
    STPO (Selective-Turn Policy Optimization)

    - Sample N candidate answers (round-1)
    - Select K answers per prompt by answer-selection rewards
    - For each selected answer, sample N/K confidence completions (round-2)
    """

    _tag_names = ["trl", "stpo"]

    def __init__(
        self,
        model: Union[str, mtpo.PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        answer_selection_reward_funcs: Union[RewardFunc, list[RewardFunc]],
        answer_selection_reward_weights: Optional[list[float]] = None,
        answer_reward_funcs: Optional[Union[RewardFunc, list[RewardFunc]]] = None,
        answer_reward_weights: Optional[list[float]] = None,
        metric_reward_funcs: Optional[Union[RewardFunc, list[RewardFunc]]] = None,
        args: mtpo.GRPOConfig = None,
        train_dataset: Optional[Union[mtpo.Dataset, mtpo.IterableDataset]] = None,
        eval_dataset: Optional[
            Union[
                mtpo.Dataset,
                mtpo.IterableDataset,
                dict[str, Union[mtpo.Dataset, mtpo.IterableDataset]],
            ]
        ] = None,
        processing_class: Optional[mtpo.PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[mtpo.PreTrainedTokenizerBase, list[mtpo.PreTrainedTokenizerBase]]
        ] = None,
        answer_selection_reward_processing_classes: Optional[
            Union[mtpo.PreTrainedTokenizerBase, list[mtpo.PreTrainedTokenizerBase]]
        ] = None,
        answer_selection_format_pattern: str = "ta",
        answer_selection_strategy: str = "topk",
        answer_selection_balanced_correct_fraction: float = 0.5,
        callbacks: Optional[list[mtpo.TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer],
            Optional[torch.optim.lr_scheduler.LambdaLR],
        ] = (None, None),
    ):
        if answer_selection_reward_funcs is None:
            raise ValueError("STPO requires answer_selection_reward_funcs.")
        if not isinstance(answer_selection_reward_funcs, list):
            answer_selection_reward_funcs = [answer_selection_reward_funcs]

        self.answer_selection_reward_funcs = list(answer_selection_reward_funcs)
        if answer_selection_reward_weights is not None:
            if len(answer_selection_reward_weights) != len(
                self.answer_selection_reward_funcs
            ):
                raise ValueError(
                    "answer_selection_reward_weights must match answer_selection_reward_funcs length "
                    f"({len(answer_selection_reward_weights)} vs {len(self.answer_selection_reward_funcs)})."
                )
            self.answer_selection_reward_weights = torch.tensor(
                answer_selection_reward_weights, dtype=torch.float32
            )
        else:
            self.answer_selection_reward_weights = torch.ones(
                len(self.answer_selection_reward_funcs), dtype=torch.float32
            ) / max(1, len(self.answer_selection_reward_funcs))

        self.answer_selection_format_pattern = answer_selection_format_pattern
        self.answer_selection_strategy = str(answer_selection_strategy or "topk")
        self.answer_selection_balanced_correct_fraction = float(
            answer_selection_balanced_correct_fraction
        )

        # Answer-training rewards (applied on round-1 answer-only candidates).
        if answer_reward_funcs is None:
            answer_reward_funcs = self.answer_selection_reward_funcs
        if not isinstance(answer_reward_funcs, list):
            answer_reward_funcs = [answer_reward_funcs]
        self.answer_reward_funcs = list(answer_reward_funcs)
        if answer_reward_weights is not None:
            if len(answer_reward_weights) != len(self.answer_reward_funcs):
                raise ValueError(
                    "answer_reward_weights must match answer_reward_funcs length "
                    f"({len(answer_reward_weights)} vs {len(self.answer_reward_funcs)})."
                )
            self.answer_reward_weights = torch.tensor(
                answer_reward_weights, dtype=torch.float32
            )
        else:
            self.answer_reward_weights = torch.ones(
                len(self.answer_reward_funcs), dtype=torch.float32
            ) / max(1, len(self.answer_reward_funcs))

        # Metric-only rewards (computed for logging only; do NOT affect training).
        if metric_reward_funcs is None:
            metric_reward_funcs = []
        if not isinstance(metric_reward_funcs, list):
            metric_reward_funcs = [metric_reward_funcs]
        self.metric_reward_funcs = list(metric_reward_funcs)

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        model_init_kwargs = self.args.model_init_kwargs or {}

        self.answer_reward_func_names: list[str] = []
        for reward_func in self.answer_reward_funcs:
            if isinstance(reward_func, mtpo.nn.Module):
                self.answer_reward_func_names.append(
                    reward_func.config._name_or_path.split("/")[-1]
                )
            else:
                try:
                    self.answer_reward_func_names.append(reward_func.__name__)
                except Exception:
                    self.answer_reward_func_names.append(reward_func.func.__name__)

        self.metric_reward_func_names: list[str] = []
        for reward_func in self.metric_reward_funcs:
            if isinstance(reward_func, mtpo.nn.Module):
                self.metric_reward_func_names.append(
                    reward_func.config._name_or_path.split("/")[-1]
                )
            else:
                try:
                    self.metric_reward_func_names.append(reward_func.__name__)
                except Exception:
                    self.metric_reward_func_names.append(reward_func.func.__name__)

        # Keep reward metric naming consistent with MTPO: `rewards/<reward_func_name>`.
        # Since STPO logs answer-side and confidence-side rewards on different sample sets,
        # we require reward names to be unambiguous across stages.
        overlap = set(self.answer_reward_func_names) & set(self.reward_func_names)
        if overlap:
            raise ValueError(
                "STPO requires answer_reward_funcs and confidence_reward_funcs to have distinct reward function "
                "names for logging consistency. Overlap: " + ", ".join(sorted(overlap))
            )
        overlap = set(self.answer_reward_func_names) & set(
            self.metric_reward_func_names
        )
        if overlap:
            raise ValueError(
                "STPO requires answer_reward_funcs and metric_reward_funcs to have distinct reward function "
                "names for logging consistency. Overlap: " + ", ".join(sorted(overlap))
            )

        self.answer_selection_reward_func_names: list[str] = []
        for i, reward_func in enumerate(self.answer_selection_reward_funcs):
            if isinstance(reward_func, str):
                self.answer_selection_reward_funcs[i] = (
                    mtpo.AutoModelForSequenceClassification.from_pretrained(
                        reward_func, num_labels=1, **model_init_kwargs
                    )
                )
            reward_func = self.answer_selection_reward_funcs[i]
            if isinstance(reward_func, mtpo.nn.Module):
                self.answer_selection_reward_func_names.append(
                    reward_func.config._name_or_path.split("/")[-1]
                )
            else:
                try:
                    self.answer_selection_reward_func_names.append(reward_func.__name__)
                except Exception:
                    self.answer_selection_reward_func_names.append(
                        reward_func.func.__name__
                    )

        if answer_selection_reward_processing_classes is None:
            answer_selection_reward_processing_classes = [None] * len(
                self.answer_selection_reward_funcs
            )
        elif not isinstance(answer_selection_reward_processing_classes, list):
            answer_selection_reward_processing_classes = [
                answer_selection_reward_processing_classes
            ]
        else:
            if len(answer_selection_reward_processing_classes) != len(
                self.answer_selection_reward_funcs
            ):
                raise ValueError(
                    "The number of answer selection reward processing classes must match "
                    "the number of answer selection reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(
                answer_selection_reward_processing_classes,
                self.answer_selection_reward_funcs,
            )
        ):
            if isinstance(reward_func, mtpo.PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = mtpo.AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                answer_selection_reward_processing_classes[i] = reward_processing_class
        self.answer_selection_reward_processing_classes = (
            answer_selection_reward_processing_classes
        )

        for i, reward_func in enumerate(self.answer_selection_reward_funcs):
            if isinstance(reward_func, mtpo.PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.answer_selection_reward_funcs[i] = mtpo.prepare_deepspeed(
                        reward_func, self.accelerator
                    )
                else:
                    self.answer_selection_reward_funcs[i] = (
                        self.accelerator.prepare_model(
                            reward_func, evaluation_mode=True, device_placement=True
                        )
                    )

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]], eval=False
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            mtpo.maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = mtpo.Trainer._prepare_inputs(self, prompt_inputs)
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

        def _as_list(tokens: Any) -> list[int]:
            if isinstance(tokens, list):
                return tokens
            if isinstance(tokens, tuple):
                return list(tokens)
            if isinstance(tokens, torch.Tensor):
                return [int(x) for x in tokens.tolist()]
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

        # STPO batches contain two parts per prompt:
        # - Round-1: C candidate answers (answer-only samples)
        # - Round-2: G*H confidence samples over selected answers, with H=C//G (so total is 2C)
        C = int(getattr(self.args, "num_answer_candidates", 0))
        G = int(getattr(self.args, "num_answer_selected", 0))
        H = int(getattr(self.args, "num_confidence_generations", 0))
        if C < 1 or G < 1:
            raise ValueError(
                "STPO requires args.num_answer_candidates >= 1 and args.num_answer_selected >= 1 "
                f"(got {C}, {G})."
            )
        if C % G != 0:
            raise ValueError(
                "STPO requires num_answer_candidates % num_answer_selected == 0 "
                f"(got {C} % {G})."
            )
        if H <= 0:
            H = C // G
        if H != C // G:
            raise ValueError(
                "STPO requires num_confidence_generations == num_answer_candidates // num_answer_selected "
                f"(got H={H} vs {C}//{G}={C // G})."
            )
        total_per_prompt = C + (G * H)
        if int(self.num_generations) != total_per_prompt:
            raise ValueError(
                "STPO requires args.num_generations == num_answer_candidates + num_answer_candidates "
                f"(got {int(self.num_generations)} vs {total_per_prompt})."
            )
        if len(inputs) % total_per_prompt != 0:
            raise ValueError(
                f"Batch size ({len(inputs)}) must be divisible by num_generations ({total_per_prompt})."
            )

        group_start_indices = list(range(0, len(inputs), total_per_prompt))
        group_prompts: list[mtpo.TokensPrompt] = [
            {"prompt_token_ids": prompt_token_ids[i]} for i in group_start_indices
        ]

        # -------------------------
        # Round 1: sample C candidate answers per prompt
        # -------------------------
        sampling_params_answer = mtpo.SamplingParams(
            n=C,
            temperature=self.temperature,
            max_tokens=self.max_answer_length,
            stop=self._answer_stop_str_variants,
            include_stop_str_in_output=True,
        )
        with mtpo.profiling_context(self, "vLLM.generate_candidate_answers"):
            answer_outputs = self.llm.generate(
                group_prompts, sampling_params=sampling_params_answer, use_tqdm=False
            )

        candidate_answer_ids: list[list[list[int]]] = []
        candidate_valid: list[list[bool]] = []
        answer_close_found_flags: list[bool] = []
        answer_finish_stop_flags: list[bool] = []
        for outputs in answer_outputs:
            group_ids: list[list[int]] = []
            group_valid: list[bool] = []
            for output in outputs.outputs:
                tok = _strip_eos(output.token_ids)
                tok, found_close = _truncate_to_last_closed_tag(
                    tok, "<answer>", "</answer>"
                )
                finish_reason = getattr(output, "finish_reason", None)
                answer_close_found_flags.append(bool(found_close))
                answer_finish_stop_flags.append(finish_reason == "stop")
                group_ids.append(tok)
                group_valid.append(bool(found_close))
            if len(group_ids) != C:
                raise ValueError(
                    f"Expected {C} candidate answers per prompt, got {len(group_ids)}."
                )
            candidate_answer_ids.append(group_ids)
            candidate_valid.append(group_valid)

        if answer_close_found_flags:
            close_ratio = float(sum(answer_close_found_flags)) / len(
                answer_close_found_flags
            )
            self._metrics[mode]["stpo/format_answer_close_ratio"].append(close_ratio)
        if answer_finish_stop_flags:
            self._metrics[mode]["stpo/answer_finish_reason_stop_ratio"].append(
                float(sum(answer_finish_stop_flags)) / len(answer_finish_stop_flags)
            )

        # -------------------------
        # Select top-G answers per prompt by answer-selection rewards
        # -------------------------
        flat_candidate_ids: list[list[int]] = [
            ids for group in candidate_answer_ids for ids in group
        ]
        flat_candidate_text = self.processing_class.batch_decode(
            flat_candidate_ids, skip_special_tokens=True
        )

        # Build prompts/completions aligned to flat_candidate_* (C per prompt group).
        selection_prompts = []
        for idx in group_start_indices:
            selection_prompts.extend([inputs[idx]["prompt"]] * C)

        if mtpo.is_conversational(inputs[0]):
            fixed_prompts = []
            selection_completions = []
            for prompt, completion in zip(selection_prompts, flat_candidate_text):
                if prompt and prompt[-1]["role"] == "assistant":
                    bootstrap = prompt[-1]["content"]
                    fixed_prompts.append(prompt[:-1])
                else:
                    bootstrap = ""
                    fixed_prompts.append(prompt)
                selection_completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
            selection_prompts = fixed_prompts
        else:
            selection_completions = flat_candidate_text

        selection_rewards_per_func = torch.zeros(
            len(selection_prompts),
            len(self.answer_selection_reward_funcs),
            device=device,
        )

        # Build reward kwargs (repeat non-prompt fields C times per group).
        keys = [
            key
            for key in inputs[0]
            if key not in ["prompt", "completion", "completion_ids"]
        ]
        selection_reward_kwargs = {key: [] for key in keys}
        for idx in group_start_indices:
            for key in keys:
                selection_reward_kwargs[key].extend([inputs[idx][key]] * C)

        for i, (
            reward_func,
            reward_processing_class,
            reward_func_name,
        ) in enumerate(
            zip(
                self.answer_selection_reward_funcs,
                self.answer_selection_reward_processing_classes,
                self.answer_selection_reward_func_names,
            )
        ):
            with mtpo.profiling_context(self, f"stpo_select/{reward_func_name}"):
                if isinstance(reward_func, mtpo.nn.Module):
                    if mtpo.is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c}
                            for p, c in zip(selection_prompts, selection_completions)
                        ]
                        texts = [
                            mtpo.apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [
                            p + c
                            for p, c in zip(selection_prompts, selection_completions)
                        ]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = mtpo.Trainer._prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        selection_rewards_per_func[:, i] = reward_func(
                            **reward_inputs
                        ).logits[:, 0]
                else:
                    output_reward_func = reward_func(
                        prompts=selection_prompts,
                        completions=selection_completions,
                        completion_ids=flat_candidate_ids,
                        **selection_reward_kwargs,
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    selection_rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        selection_rewards_per_func = torch.nan_to_num(
            selection_rewards_per_func, nan=0.0
        )
        sel_w = self.answer_selection_reward_weights.to(
            selection_rewards_per_func.device
        )
        selection_scores = (selection_rewards_per_func * sel_w.unsqueeze(0)).sum(dim=1)

        flat_valid = torch.tensor(
            [v for group in candidate_valid for v in group],
            dtype=torch.bool,
            device=selection_scores.device,
        )
        valid_grouped = flat_valid.view(-1, C)
        selection_scores_grouped = selection_scores.view(-1, C)

        # -------------------------
        # Answer-training rewards (Round-1 candidates)
        # -------------------------
        if self.answer_reward_funcs == self.answer_selection_reward_funcs:
            answer_rewards_per_func = selection_rewards_per_func
        else:
            answer_rewards_per_func = torch.zeros(
                len(selection_prompts),
                len(self.answer_reward_funcs),
                device=device,
            )
            for i, (reward_func, reward_func_name) in enumerate(
                zip(self.answer_reward_funcs, self.answer_reward_func_names)
            ):
                if isinstance(reward_func, mtpo.nn.Module):
                    raise ValueError(
                        "answer_reward_funcs does not support model-based reward functions yet; "
                        "please use custom python reward functions."
                    )
                with mtpo.profiling_context(self, f"stpo_answer/{reward_func_name}"):
                    output_reward_func = reward_func(
                        prompts=selection_prompts,
                        completions=selection_completions,
                        completion_ids=flat_candidate_ids,
                        **selection_reward_kwargs,
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    answer_rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        answer_rewards_per_func = torch.nan_to_num(answer_rewards_per_func, nan=0.0)
        if answer_rewards_per_func.numel() > 0:
            gathered_answer_rewards_per_func = mtpo.gather(answer_rewards_per_func)
            for i, reward_func_name in enumerate(self.answer_reward_func_names):
                reward_values = gathered_answer_rewards_per_func[:, i]
                self._metrics[mode][f"rewards/{reward_func_name}"].append(
                    torch.nanmean(reward_values).item()
                )
                self._metrics[mode][f"rewards/{reward_func_name}/batch_std"].append(
                    mtpo.nanstd(reward_values).item()
                )

                group_std_rewards = 0.0
                group_size = int(C)
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
                            group_std_rewards = (
                                torch.sqrt(var)[has_enough].mean().item()
                            )
                self._metrics[mode][f"rewards/{reward_func_name}/group_std"].append(
                    group_std_rewards
                )

        ans_w = self.answer_reward_weights.to(answer_rewards_per_func.device)
        answer_scores = (answer_rewards_per_func * ans_w.unsqueeze(0)).sum(dim=1)
        answer_scores_grouped = answer_scores.view(-1, C)

        # Prefer selecting only candidates that are "format-correct" for the answer segment.
        # This is stricter than `candidate_valid` (which only checks for a closed </answer> tag).
        format_idx: Optional[int] = None
        for j, name in enumerate(self.answer_selection_reward_func_names):
            n = str(name)
            if "format_answer_segment" in n or n == "format_reward":
                format_idx = j
                break
        if format_idx is None:
            format_ok_grouped = valid_grouped
        else:
            format_ok_flat = selection_rewards_per_func[:, format_idx] > 0.5
            format_ok_grouped = format_ok_flat.view(-1, C) & valid_grouped

        if format_ok_grouped.numel() > 0:
            ans_fmt_ratio = format_ok_grouped.float().mean().item()
            self._metrics[mode]["stpo/format_answer_segment_ratio"].append(
                ans_fmt_ratio
            )

        # Answer baseline: computed over all candidate answers (C) per prompt group,
        # but centered/scaled w.r.t. format-correct candidates (so we keep signal when format collapses).
        valid_f = format_ok_grouped.to(answer_scores_grouped.dtype)
        valid_counts = valid_f.sum(dim=1, keepdim=True)
        denom = valid_counts.clamp(min=1.0)
        ans_mean_all = (answer_scores_grouped * valid_f).sum(
            dim=1, keepdim=True
        ) / denom
        diff_all = answer_scores_grouped - ans_mean_all
        var_all = ((diff_all * valid_f) ** 2).sum(dim=1, keepdim=True) / denom
        ans_std_all = torch.sqrt(var_all)
        answer_adv_candidates = diff_all
        if self.scale_rewards:
            answer_adv_candidates = answer_adv_candidates / (ans_std_all + 1e-4)
        # Do NOT zero-out invalid candidates here.
        # If we mask them to 0 advantage, the model receives no gradient signal to recover
        # when answer formatting starts collapsing (close-tag ratio drops), and metrics will
        # keep drifting down. Keeping their (typically negative) advantage penalizes invalid
        # answers relative to valid ones, mirroring MTPO's "invalid -> low reward" behavior.
        #
        # If a prompt group has *no* valid candidates at all, then all selection scores are
        # usually identical (often 0) and diff_all provides no signal. Add a small constant
        # negative advantage in that edge case to encourage exploration away from the
        # degenerate regime.
        no_valid_group = valid_counts.squeeze(1) < 0.5
        if no_valid_group.any():
            answer_adv_candidates[no_valid_group] = (
                answer_adv_candidates[no_valid_group] - 1.0
            )

        def _max_divisor_leq(total: int, limit: int) -> int:
            if limit <= 0:
                return 0
            for d in range(int(limit), 0, -1):
                if total % d == 0:
                    return d
            return 1

        strategy = self.answer_selection_strategy.lower()
        scores_cpu = selection_scores_grouped.detach().to("cpu")
        format_ok_cpu = format_ok_grouped.detach().to("cpu")
        close_ok_cpu = valid_grouped.detach().to("cpu")

        gen = torch.Generator()
        gen.manual_seed(int(self.args.seed) + int(self.state.global_step))

        g_eff_per_group: list[int] = []
        h_eff_per_group: list[int] = []
        selected_candidate_indices_per_group: list[list[int]] = []
        selected_format_ok_flags: list[bool] = []

        correct_mask_cpu: Optional[torch.Tensor] = None
        frac = float(self.answer_selection_balanced_correct_fraction)
        frac = max(0.0, min(1.0, frac))
        if strategy == "balanced_accuracy":
            if "answer" not in selection_reward_kwargs:
                raise ValueError(
                    "balanced_accuracy selection requires an 'answer' column in the dataset."
                )
            correctness = accuracy_reward(
                self.answer_selection_format_pattern or "ta",
                selection_completions,
                selection_reward_kwargs["answer"],
                source=selection_reward_kwargs.get("source"),
            )
            correctness_grouped = torch.tensor(
                [float(x) for x in correctness], dtype=torch.float32
            ).view(-1, C)
            correct_mask_cpu = (correctness_grouped >= 0.5).to(torch.bool)

        for group_i in range(scores_cpu.size(0)):
            eligible_idx = torch.nonzero(
                format_ok_cpu[group_i], as_tuple=False
            ).squeeze(1)
            eligible_count = int(eligible_idx.numel())
            limit = min(G, eligible_count)
            g_eff = _max_divisor_leq(C, limit)
            if g_eff < 1:
                g_eff = 1
            h_eff = C // g_eff

            chosen_idx: torch.Tensor
            if eligible_count >= 1 and g_eff <= eligible_count:
                if strategy == "topk":
                    scores_row = scores_cpu[group_i][eligible_idx]
                    top = torch.topk(
                        scores_row, k=g_eff, largest=True, sorted=True
                    ).indices
                    chosen_idx = eligible_idx[top]
                elif strategy == "random":
                    chosen_idx = eligible_idx[
                        torch.randperm(eligible_count, generator=gen)[:g_eff]
                    ]
                elif strategy == "balanced_accuracy":
                    assert correct_mask_cpu is not None
                    scores_row_all = scores_cpu[group_i]
                    eligible_row = format_ok_cpu[group_i]
                    correct_row = correct_mask_cpu[group_i] & eligible_row
                    incorrect_row = (~correct_mask_cpu[group_i]) & eligible_row

                    target_correct = int(round(g_eff * frac))
                    target_correct = max(0, min(g_eff, target_correct))
                    target_incorrect = g_eff - target_correct

                    selected_parts = []
                    k_c = min(target_correct, int(correct_row.sum().item()))
                    if k_c > 0:
                        s = scores_row_all.clone()
                        s[~correct_row] = -1e9
                        selected_parts.append(
                            torch.topk(s, k=k_c, largest=True, sorted=True).indices
                        )
                    k_i = min(target_incorrect, int(incorrect_row.sum().item()))
                    if k_i > 0:
                        s = scores_row_all.clone()
                        s[~incorrect_row] = -1e9
                        selected_parts.append(
                            torch.topk(s, k=k_i, largest=True, sorted=True).indices
                        )

                    chosen = (
                        torch.cat(selected_parts, dim=0)
                        if selected_parts
                        else torch.empty(0, dtype=torch.long)
                    )
                    if chosen.numel() < g_eff:
                        s = scores_row_all.clone()
                        s[~eligible_row] = -1e9
                        if chosen.numel() > 0:
                            s[chosen] = -1e9
                        fill = torch.topk(
                            s, k=g_eff - chosen.numel(), largest=True, sorted=True
                        ).indices
                        chosen = torch.cat([chosen, fill], dim=0)
                    chosen_idx = chosen
                else:
                    raise ValueError(
                        "Unknown answer_selection_strategy: "
                        f"{self.answer_selection_strategy}. Expected one of: topk, random, balanced_accuracy."
                    )
            else:
                # If there are no format-correct candidates, still pick a single answer to
                # keep the rollout shape consistent, but mark it invalid for confidence rewards.
                # Prefer answers with a closed </answer> tag to make round-2 generation well-posed.
                pool = torch.nonzero(close_ok_cpu[group_i], as_tuple=False).squeeze(1)
                if pool.numel() == 0:
                    pool = torch.arange(C, dtype=torch.long)
                if strategy == "topk":
                    scores_row = scores_cpu[group_i][pool]
                    chosen_idx = pool[
                        torch.topk(scores_row, k=1, largest=True, sorted=True).indices
                    ]
                else:
                    chosen_idx = pool[
                        torch.randperm(int(pool.numel()), generator=gen)[:1]
                    ]
                g_eff = 1
                h_eff = C

            chosen_list = [int(x) for x in chosen_idx.tolist()]
            selected_candidate_indices_per_group.append(chosen_list)
            g_eff_per_group.append(int(g_eff))
            h_eff_per_group.append(int(h_eff))
            for cand_idx in chosen_list:
                selected_format_ok_flags.append(bool(format_ok_cpu[group_i, cand_idx]))

        if g_eff_per_group:
            self._metrics[mode]["stpo/num_answer_selected_eff"].append(
                float(sum(g_eff_per_group)) / len(g_eff_per_group)
            )
        if h_eff_per_group:
            self._metrics[mode]["stpo/num_confidence_generations_eff"].append(
                float(sum(h_eff_per_group)) / len(h_eff_per_group)
            )
        if selected_format_ok_flags:
            sel_valid_ratio = float(
                sum(1.0 for x in selected_format_ok_flags if x)
            ) / len(selected_format_ok_flags)
            self._metrics[mode]["stpo/format_answer_selected_ratio"].append(
                sel_valid_ratio
            )

        # Answer advantages are computed over all C candidate answers. We will apply them on the
        # round-1 (answer-only) samples for all candidates; round-2 samples only carry confidence loss.

        # -------------------------
        # Round 2: sample H confidences per selected answer
        # -------------------------
        confidence_slot_prompts: list[mtpo.TokensPrompt] = []
        confidence_slot_meta: list[tuple[int, int, int]] = []
        selected_answer_ids: list[list[list[int]]] = []
        selected_answer_format_ok: list[list[bool]] = []
        for group_i, start_idx in enumerate(group_start_indices):
            group_selected_ids: list[list[int]] = []
            group_selected_ok: list[bool] = []
            g_eff = g_eff_per_group[group_i]
            h_eff = h_eff_per_group[group_i]
            for g in range(g_eff):
                cand_idx = int(selected_candidate_indices_per_group[group_i][g])
                ans_ids = candidate_answer_ids[group_i][cand_idx]
                group_selected_ids.append(ans_ids)
                ok = bool(format_ok_cpu[group_i, cand_idx])
                group_selected_ok.append(ok)
                prefix = prompt_token_ids[start_idx] + ans_ids
                confidence_slot_prompts.append({"prompt_token_ids": prefix})
                confidence_slot_meta.append((group_i, g, h_eff))
            selected_answer_ids.append(group_selected_ids)
            selected_answer_format_ok.append(group_selected_ok)

        # Generate confidence samples. vLLM's `n` is global per generate() call, so bucket by H_eff.
        confidence_ids_grouped: list[list[list[list[int]]]] = [
            [None for _ in range(len(selected_answer_ids[g]))]
            for g in range(len(selected_answer_ids))
        ]
        if confidence_slot_prompts:
            buckets: dict[int, list[int]] = {}
            for slot_i, (_, __, h_eff) in enumerate(confidence_slot_meta):
                buckets.setdefault(int(h_eff), []).append(slot_i)

            with mtpo.profiling_context(self, "vLLM.generate_confidence"):
                for h_eff, slot_indices in buckets.items():
                    sampling_params_conf = mtpo.SamplingParams(
                        n=int(h_eff),
                        temperature=self.temperature,
                        max_tokens=self.max_confidence_length,
                        stop=self._confidence_stop_str_variants,
                        include_stop_str_in_output=True,
                    )
                    outs = self.llm.generate(
                        [confidence_slot_prompts[i] for i in slot_indices],
                        sampling_params=sampling_params_conf,
                        use_tqdm=False,
                    )
                    if len(outs) != len(slot_indices):
                        raise ValueError(
                            "Unexpected vLLM output length for confidence generation "
                            f"({len(outs)} vs {len(slot_indices)})."
                        )
                    for slot_i, outputs in zip(slot_indices, outs):
                        group_i, g, expected_h = confidence_slot_meta[slot_i]
                        if int(expected_h) != int(h_eff):
                            raise ValueError("Internal error: H_eff bucket mismatch.")
                        group_conf: list[list[int]] = []
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
                                has_text_stop = decoded.rstrip().endswith(
                                    self.confidence_stop_str
                                )
                            if (
                                (not found_stop)
                                and (not has_text_stop)
                                and finish_reason == "stop"
                            ):
                                pass
                            if self.processing_class.eos_token_id is not None and (
                                len(tok) == 0
                                or tok[-1] != self.processing_class.eos_token_id
                            ):
                                tok = tok + [self.processing_class.eos_token_id]
                            group_conf.append(tok)
                        if len(group_conf) != int(h_eff):
                            raise ValueError(
                                f"Expected {h_eff} confidences per selected answer, got {len(group_conf)}."
                            )
                        confidence_ids_grouped[group_i][g] = group_conf

        # Ensure all confidence slots are filled.
        for group_i in range(len(group_start_indices)):
            for g in range(len(selected_answer_ids[group_i])):
                if confidence_ids_grouped[group_i][g] is None:
                    raise ValueError(
                        "Missing confidence outputs for a selected answer."
                    )

        if mode == "train":
            self.llm.sleep(level=1)
            self.vllm_sleeping = True
            self.accelerator.wait_for_everyone()

        # -------------------------
        # Combine into one completion sequence:
        # - First C samples: candidate answers (answer-only)
        # - Next C samples: selected answers + confidence (answer + confidence)
        # -------------------------
        combined_completion_tensors: list[torch.Tensor] = []
        stage1_lens: list[int] = []
        stage2_lens: list[int] = []
        valid_mask_local = torch.zeros(len(inputs), dtype=torch.float32, device=device)

        sample_idx = 0
        for group_i in range(len(group_start_indices)):
            # Round 1: all candidate answers (answer-only) contribute to answer loss.
            for c in range(C):
                stage1 = candidate_answer_ids[group_i][c]
                if self.processing_class.eos_token_id is not None and (
                    len(stage1) == 0 or stage1[-1] != self.processing_class.eos_token_id
                ):
                    stage1 = stage1 + [self.processing_class.eos_token_id]
                # IMPORTANT: do not apply answer loss on the terminal EOS for answer-only samples.
                # Otherwise the policy is reinforced to end right after </answer>, which collapses the
                # round-2 confidence rollout and makes full-completion rewards (acc/brier/format) drop.
                stage1_len_for_mask = len(stage1)
                if (
                    stage1_len_for_mask > 0
                    and self.processing_class.eos_token_id is not None
                    and stage1[-1] == self.processing_class.eos_token_id
                ):
                    stage1_len_for_mask = max(0, stage1_len_for_mask - 1)
                is_valid = candidate_valid[group_i][c]
                combined_completion_tensors.append(torch.tensor(stage1, device=device))
                stage1_lens.append(stage1_len_for_mask)
                stage2_lens.append(0)
                valid_mask_local[sample_idx] = 1.0 if is_valid else 0.0
                sample_idx += 1

            # Round 2: confidence rollouts for selected answers contribute to confidence loss.
            g_eff = g_eff_per_group[group_i]
            h_eff = h_eff_per_group[group_i]
            for g in range(g_eff):
                stage1 = selected_answer_ids[group_i][g]
                is_valid = selected_answer_format_ok[group_i][g]
                for h in range(h_eff):
                    stage2 = confidence_ids_grouped[group_i][g][h]
                    combined = stage1 + stage2
                    combined_completion_tensors.append(
                        torch.tensor(combined, device=device)
                    )
                    stage1_lens.append(len(stage1))
                    stage2_lens.append(len(stage2))
                    valid_mask_local[sample_idx] = 1.0 if is_valid else 0.0
                    sample_idx += 1
        if sample_idx != len(inputs):
            raise ValueError(
                f"Expected to build {len(inputs)} completions, got {sample_idx}."
            )

        completion_ids = mtpo.pad(
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
            within_group = i % total_per_prompt
            is_candidate_sample = within_group < C
            if a_len > 0 and is_candidate_sample:
                answer_token_mask[i, :a_len] = 1.0
            if c_len > 0:
                confidence_token_mask[i, a_len : a_len + c_len] = 1.0

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]

        truncated_completions = None
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        answer_token_mask = answer_token_mask * completion_mask
        confidence_token_mask = confidence_token_mask * completion_mask

        completion_lengths = completion_mask.sum(1).to(torch.float32)
        answer_lengths = torch.tensor(stage1_lens, device=device, dtype=torch.float32)
        answer_lengths = torch.minimum(answer_lengths, completion_lengths)
        confidence_lengths = (completion_lengths - answer_lengths).clamp(min=0)
        split_ratios = torch.full_like(answer_lengths, float("nan"))
        invalid_samples = valid_mask_local < 0.5
        answer_lengths = answer_lengths.masked_fill(invalid_samples, float("nan"))
        confidence_lengths = confidence_lengths.masked_fill(
            invalid_samples, float("nan")
        )
        valid_split = (~invalid_samples) & (completion_lengths > 0)
        split_ratios = torch.where(
            valid_split,
            answer_lengths / completion_lengths.clamp(min=1.0),
            split_ratios,
        )
        if truncated_completions is not None:
            answer_lengths = answer_lengths.masked_fill(
                truncated_completions, float("nan")
            )
            confidence_lengths = confidence_lengths.masked_fill(
                truncated_completions, float("nan")
            )
            split_ratios = split_ratios.masked_fill(truncated_completions, float("nan"))

        # Candidate samples have no confidence span; exclude them from confidence-span metrics.
        sample_idx = torch.arange(len(inputs), device=device)
        is_candidate_sample = (sample_idx % total_per_prompt) < C
        confidence_lengths = confidence_lengths.masked_fill(
            is_candidate_sample, float("nan")
        )
        split_ratios = split_ratios.masked_fill(is_candidate_sample, float("nan"))

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
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

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if mtpo.is_conversational(inputs[0]):
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
        # Compute training rewards only on round-2 confidence samples (these contain <confidence>).
        idx = torch.arange(len(inputs), device=device)
        within_group = idx % total_per_prompt
        is_confidence_sample = within_group >= C
        conf_indices = torch.nonzero(is_confidence_sample, as_tuple=False).squeeze(1)
        conf_indices_cpu = conf_indices.detach().to("cpu").tolist()

        reward_prompts = [prompts[i] for i in conf_indices_cpu]
        reward_completions = [completions[i] for i in conf_indices_cpu]
        reward_completion_ids_list = [completion_ids_list[i] for i in conf_indices_cpu]
        reward_prompts_text = [prompts_text[i] for i in conf_indices_cpu]
        reward_completions_text = [completions_text[i] for i in conf_indices_cpu]
        reward_valid_mask_local = valid_mask_local[conf_indices]

        rewards_per_func = torch.zeros(
            len(reward_prompts), len(self.reward_funcs), device=device
        )
        keys = [
            key
            for key in inputs[0]
            if key not in ["prompt", "completion", "completion_ids"]
        ]
        reward_kwargs = {
            key: [inputs[i][key] for i in conf_indices_cpu] for key in keys
        }

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
            )
        ):
            with mtpo.profiling_context(self, reward_func_name):
                if isinstance(reward_func, mtpo.nn.Module):
                    if mtpo.is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c}
                            for p, c in zip(reward_prompts, reward_completions)
                        ]
                        texts = [
                            mtpo.apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [
                            p + c for p, c in zip(reward_prompts, reward_completions)
                        ]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = mtpo.Trainer._prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                            :, 0
                        ]
                else:
                    output_reward_func = reward_func(
                        prompts=reward_prompts,
                        completions=reward_completions,
                        completion_ids=reward_completion_ids_list,
                        **reward_kwargs,
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        metric_rewards_per_func = None
        if self.metric_reward_funcs:
            metric_rewards_per_func = torch.zeros(
                len(reward_prompts), len(self.metric_reward_funcs), device=device
            )
            for i, (reward_func, reward_func_name) in enumerate(
                zip(self.metric_reward_funcs, self.metric_reward_func_names)
            ):
                if isinstance(reward_func, mtpo.nn.Module):
                    raise ValueError(
                        "metric_reward_funcs does not support model-based reward functions yet; "
                        "please use custom python reward functions."
                    )
                with mtpo.profiling_context(self, f"stpo_metric/{reward_func_name}"):
                    output_reward_func = reward_func(
                        prompts=reward_prompts,
                        completions=reward_completions,
                        completion_ids=reward_completion_ids_list,
                        **reward_kwargs,
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    metric_rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        rewards_per_func = mtpo.gather(rewards_per_func)
        rewards_per_func = torch.nan_to_num(rewards_per_func, nan=0.0)
        if metric_rewards_per_func is not None:
            metric_rewards_per_func = mtpo.gather(metric_rewards_per_func)
            metric_rewards_per_func = torch.nan_to_num(metric_rewards_per_func, nan=0.0)
        valid_mask = mtpo.gather(reward_valid_mask_local)

        if metric_rewards_per_func is not None and metric_rewards_per_func.numel() > 0:
            already_logged = set(self.reward_func_names) | set(
                self.answer_reward_func_names
            )
            for i, reward_func_name in enumerate(self.metric_reward_func_names):
                if reward_func_name in already_logged:
                    continue
                reward_values = metric_rewards_per_func[:, i]
                self._metrics[mode][f"rewards/{reward_func_name}"].append(
                    torch.nanmean(reward_values).item()
                )
                self._metrics[mode][f"rewards/{reward_func_name}/batch_std"].append(
                    mtpo.nanstd(reward_values).item()
                )

                group_std_rewards = 0.0
                group_size = int(C)
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
                            group_std_rewards = (
                                torch.sqrt(var)[has_enough].mean().item()
                            )
                self._metrics[mode][f"rewards/{reward_func_name}/group_std"].append(
                    group_std_rewards
                )

        # Extra STPO metrics (computed on Round-2 samples before masking by answer validity).
        def _find_reward_idx(names: list[str], needle: str) -> Optional[int]:
            needle = str(needle).lower()
            for j, name in enumerate(names):
                if needle in str(name).lower():
                    return j
            return None

        if rewards_per_func.numel() > 0:
            valid_bool = valid_mask.to(torch.bool)

            def _lookup_metric(needle: str) -> Optional[torch.Tensor]:
                idx = _find_reward_idx(self.reward_func_names, needle)
                if idx is not None:
                    return rewards_per_func[:, idx]
                if metric_rewards_per_func is None:
                    return None
                midx = _find_reward_idx(self.metric_reward_func_names, needle)
                if midx is None:
                    return None
                return metric_rewards_per_func[:, midx]

            conf_fmt = _lookup_metric("format_confidence_segment")
            if conf_fmt is not None:
                self._metrics[mode]["stpo/format_confidence_segment_ratio"].append(
                    conf_fmt.float().mean().item()
                )
                if valid_bool.any():
                    self._metrics[mode][
                        "stpo/format_confidence_segment_ratio_valid_answer"
                    ].append(conf_fmt[valid_bool].float().mean().item())

            mean_conf = _lookup_metric("mean_confidence")
            if mean_conf is not None:
                self._metrics[mode]["stpo/mean_confidence"].append(
                    mean_conf.float().mean().item()
                )
                if valid_bool.any():
                    self._metrics[mode]["stpo/mean_confidence_valid_answer"].append(
                        mean_conf[valid_bool].float().mean().item()
                    )

            one_zero = _lookup_metric("confidence_one_or_zero")
            if one_zero is not None:
                self._metrics[mode]["stpo/confidence_one_or_zero_ratio"].append(
                    one_zero.float().mean().item()
                )
                if valid_bool.any():
                    self._metrics[mode][
                        "stpo/confidence_one_or_zero_ratio_valid_answer"
                    ].append(one_zero[valid_bool].float().mean().item())

        rewards_per_func = rewards_per_func * valid_mask.unsqueeze(1)
        valid_ratio = valid_mask.mean().item() if valid_mask.numel() else 0.0
        self._metrics[mode]["valid_answer_ratio"].append(valid_ratio)

        weight_vector = self.reward_weights.to(rewards_per_func.device)
        total_rewards = (rewards_per_func * weight_vector.unsqueeze(0)).sum(dim=1)

        if total_rewards.numel() % C != 0:
            raise ValueError(
                f"Total rewards length ({total_rewards.numel()}) must be divisible by C ({C})."
            )

        # For round-2, advantages should reflect rewards that *vary across confidence rollouts* for the
        # same selected answer. `total_rewards` is the weighted sum over the configured confidence-side
        # reward channels (self.reward_funcs). If you include answer-dependent channels here, they still
        # only update confidence tokens via the token mask.
        grouped_conf = total_rewards.view(-1, C)
        h_eff_local = torch.tensor(h_eff_per_group, device=device, dtype=torch.long)
        h_eff_global = mtpo.gather(h_eff_local).to(torch.long)
        if h_eff_global.numel() != grouped_conf.size(0):
            raise ValueError(
                "Internal error: H_eff gather mismatch "
                f"({h_eff_global.numel()} vs {grouped_conf.size(0)})."
            )

        confidence_adv = torch.zeros_like(grouped_conf)
        for group_i in range(grouped_conf.size(0)):
            h_eff = int(h_eff_global[group_i].item())
            if h_eff < 1 or (C % h_eff) != 0:
                raise ValueError(
                    f"Invalid H_eff={h_eff} for C={C}; expected a positive divisor."
                )
            row = grouped_conf[group_i].view(-1, h_eff)
            conf_mean = row.mean(dim=1, keepdim=True)
            conf_std = row.std(dim=1, keepdim=True, unbiased=False)
            adv = row - conf_mean
            if self.scale_rewards:
                adv = adv / (conf_std + 1e-4)
            confidence_adv[group_i] = adv.reshape(C)

        confidence_adv_flat = confidence_adv.reshape(-1)

        process_slice = slice(
            self.accelerator.process_index * len(reward_prompts),
            (self.accelerator.process_index + 1) * len(reward_prompts),
        )
        confidence_adv_local = confidence_adv_flat[process_slice]

        # Round-1 answer advantages: apply on all C candidate answers (answer-only samples).
        answer_adv_candidates_flat = answer_adv_candidates.reshape(-1)
        candidate_mask = within_group < C
        answer_adv_all = torch.zeros(len(inputs), dtype=torch.float32, device=device)
        confidence_adv_all = torch.zeros_like(answer_adv_all)
        answer_adv_all[candidate_mask] = answer_adv_candidates_flat
        confidence_adv_all[~candidate_mask] = confidence_adv_local

        advantages = (
            answer_adv_all.unsqueeze(1) * answer_token_mask
            + confidence_adv_all.unsqueeze(1) * confidence_token_mask
        )

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

        agg_answer_lengths = mtpo.gather(answer_lengths)
        agg_confidence_lengths = mtpo.gather(confidence_lengths)
        agg_split_ratios = mtpo.gather(split_ratios)
        valid_answer_lengths = agg_answer_lengths[torch.isfinite(agg_answer_lengths)]
        valid_confidence_lengths = agg_confidence_lengths[
            torch.isfinite(agg_confidence_lengths)
        ]
        valid_split_ratios = agg_split_ratios[torch.isfinite(agg_split_ratios)]
        if valid_answer_lengths.numel() > 0:
            self._metrics[mode]["spans/answer_length"].append(
                valid_answer_lengths.float().mean().item()
            )
        if valid_confidence_lengths.numel() > 0:
            self._metrics[mode]["spans/confidence_length"].append(
                valid_confidence_lengths.float().mean().item()
            )
        if valid_split_ratios.numel() > 0:
            self._metrics[mode]["spans/split_ratio"].append(
                valid_split_ratios.float().mean().item()
            )

        for i, reward_func_name in enumerate(self.reward_func_names):
            reward_values = rewards_per_func[:, i]
            self._metrics[mode][f"rewards/{reward_func_name}"].append(
                torch.nanmean(reward_values).item()
            )
            batch_std_rewards = mtpo.nanstd(reward_values).item()
            self._metrics[mode][f"rewards/{reward_func_name}/batch_std"].append(
                batch_std_rewards
            )

            group_std_rewards = 0.0
            group_size = int(C)
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
            group_size = int(C)
            if group_size > 0:
                num_groups = reward_values.numel() // group_size
                if num_groups > 0:
                    grouped = reward_values[: num_groups * group_size].view(
                        num_groups, group_size
                    )
                    answer_stds = []
                    for group_i in range(num_groups):
                        h_eff = int(h_eff_global[group_i].item())
                        if h_eff < 1 or (C % h_eff) != 0:
                            raise ValueError(
                                f"Invalid H_eff={h_eff} for C={C}; expected a positive divisor."
                            )
                        row = grouped[group_i].view(-1, h_eff)
                        # Std within each selected answer's confidence rollouts.
                        answer_stds.append(row.std(dim=1, unbiased=False).mean())
                    if answer_stds:
                        answer_std_rewards = torch.stack(answer_stds).mean().item()
            self._metrics[mode][f"rewards/{reward_func_name}/answer_std"].append(
                answer_std_rewards
            )

        grouped_total = total_rewards.view(-1, C)
        self._metrics[mode]["reward"].append(grouped_total.mean(dim=1).mean().item())
        self._metrics[mode]["reward_std"].append(grouped_total.std(dim=1).mean().item())

        if self.log_completions or self.print_completion:
            num_completions_to_log = self.args.num_completions_to_log
            gathered_prompts = mtpo.gather_object(reward_prompts_text)[
                0:num_completions_to_log
            ]
            gathered_completions = mtpo.gather_object(reward_completions_text)[
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
