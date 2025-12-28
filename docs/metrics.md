# RLCR Training Metrics

This project logs metrics through the custom trainers (`GRPO_Trainer.py`, `SMCR_Trainer.py`, `MTPO_Trainer.py`).

Notes:
- In evaluation, the trainer automatically prefixes metrics with `eval_` (e.g. `eval_reward`).
- Most metrics are averaged over the current logging window (whatever was accumulated since the last `log()` call).
- Many “span” metrics are computed only for samples where the `<answer>...</answer>` split is valid; invalid samples are
  treated as `NaN` and excluded from the mean.

## Common Metrics (All Trainers)

### Token / Length
- `num_tokens`: cumulative number of tokens seen (from `attention_mask.sum()`).
- `completions/mean_length`, `completions/min_length`, `completions/max_length`: completion token length statistics
  after masking everything after the first EOS.

### Spans (Answer vs Confidence)
All span metrics are computed on the completion tokens (not including the prompt).
- `spans/answer_length`: mean number of tokens in the answer span.
- `spans/confidence_length`: mean number of tokens in the confidence span.
- `spans/split_ratio`: mean `answer_length / (answer_length + confidence_length)`.

### Rewards
Reward functions are logged per-function (names depend on your `reward_funcs` list):
- `rewards/{reward_name}`: mean reward value for that reward function (after masking invalid samples where applicable).
- `rewards/{reward_name}/batch_std`: standard deviation of that reward over the batch.
- `rewards/{reward_name}/group_std`: average within-group standard deviation (grouped by prompt; group size depends on
  the algorithm).

Aggregate reward metrics:
- `reward`: mean total reward per prompt group.
- `reward_std`: std of total reward across generations within a prompt group (then averaged across prompts).

### Policy / Optimization
- `policy/entropy_per_token`: mean per-token entropy proxy computed as `-mean(logp)` over completion tokens.
- `policy/entropy_per_token_std`: std of that entropy proxy across samples.
- `clip_ratio/low_mean`, `clip_ratio/low_min`, `clip_ratio/high_mean`, `clip_ratio/high_max`, `clip_ratio/region_mean`:
  PPO-style clipping diagnostics based on the probability ratio `exp(logp - old_logp)`.
- `kl`: mean per-token KL proxy against the reference model (only when `beta != 0`).

## GRPO-Specific Metrics

- `advantages/standard_mean`, `advantages/standard_std`: stats of standard GRPO advantages.
- `advantages/mo_mean`, `advantages/mo_std`: stats of MO-GRPO advantages (only when enabled).
- `completions/clipped_ratio`: fraction of completions that did not terminate with EOS (clipped/truncated).
- `completions/mean_terminated_length`, `completions/min_terminated_length`, `completions/max_terminated_length`:
  length stats restricted to EOS-terminated completions.

## SMCR-Specific Metrics

- `valid_answer_ratio`: fraction of samples where the `<answer>...</answer>` split is valid.
- `confidence/correct_group_ratio`, `confidence/wrong_group_ratio`: within a prompt group, the fraction of valid
  samples that are correct/incorrect (requires an accuracy-like reward channel when enabled).
- `completions/clipped_ratio` and `completions/*_terminated_length`: same meaning as in GRPO.

Note:
- `reward/answer`, `reward/confidence`, `reward/format` are intentionally not logged; use per-function
  `rewards/{reward_name}` instead.

## MTPO-Specific Metrics

- `valid_answer_ratio`: fraction of samples whose stage-1 answer contains a closed `<answer>...</answer>` span.
- `mtpo/answer_stop_token_found_ratio`: fraction of stage-1 generations where a closed answer span was found.
- `mtpo/answer_finish_reason_stop_ratio`: fraction of stage-1 generations with vLLM `finish_reason == "stop"`.

