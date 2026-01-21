# STPO（Selective-Turn Policy Optimization）训练说明

STPO 的核心动机：**先用更大的预算采样更多 candidate answers（Round-1），再从中挑选更少的 answers 去生成 confidence（Round-2）**。

当前实现相对“旧版 STPO / MTPO”的关键点：
- **分段 format reward**：Answer 段（`<think>...</think><answer>...</answer>`）与 Confidence 段（`<analysis>...</analysis><confidence>...</confidence>`）分别用各自的 format reward 约束。
- **Format-gated selection**：进入 Round-2 继续 rollout 的 answers **只从 answer 段格式正确的 candidates 中选择**（否则无法可靠拿到 confidence 的 ground truth）。
- **动态 `G/H` 保持总 rollout 不变**：当 answer 段格式正确的 candidates 数量不足 `G_target` 时，自动把选中数降到能整除总 confidence rollout 数 `C` 的最大值，保证每个 prompt 的 Round-2 总样本数仍为 `C`。

## 1. 核心流程（两轮 + 选择）

设：
- `C = num_answer_candidates`（Round 1 rollout 的 candidate answers 数）
- `G_target = num_answer_selected`（期望选中的 answer 数）
- `G_eff`（实际选中的 answer 数，可能小于等于 `G_target`）
- `H_eff = C // G_eff`（每个 selected answer 对应的 confidence 数）
- `num_generations = 2 * C`（训练时每个 prompt 的总样本数固定为：`C` 个 answer-only + `C` 个 answer+confidence）

`G_eff` 的定义（按每个 prompt 独立计算）：
- 令 `valid_format_cnt` 为该 prompt 的 `C` 个 candidate answers 中，**answer 段 format 正确**的数量（见下文“format-gated selection”）
- `G_eff = max{ d | d <= min(G_target, valid_format_cnt) 且 C % d == 0 }`
- 若 `valid_format_cnt == 0`：退化为 `G_eff = 1, H_eff = C`（仍会 rollout `C` 个 confidence，但这些样本会被标记为 invalid，不参与 confidence 奖励/指标）

流程：
1) **Round 1（Answer candidates）**：对每个 prompt 先采样 `C` 个 `<think>...</think><answer>...</answer>`（在 `</answer>` 停止）。
2) **Select（format-gated）**：用 `answer_selection_reward_funcs` 在 answer-only completion 上打分，但**只从 answer 段 format 正确的 candidates 里选**；不足 `G_target` 时用 `G_eff/H_eff` 自动调整（由 `answer_selection_strategy` 控制 topk/random/balanced_accuracy）。
3) **Round 2（Confidence）**：对每个被选中的 answer 继续 rollout `H_eff` 次，只生成到 `</confidence>` 为止；总共 rollout `G_eff * H_eff = C` 个 confidence 样本。

用于梯度更新的样本：
- **Answer-only（Round 1）**：`C` 个（所有 candidate answers 都用于优化 answer）
- **Answer+Confidence（Round 2）**：`G_eff * H_eff = C` 个（只对被选中的 answers 继续 rollout confidence 并用于优化 confidence）

因此每个 prompt 总样本数：`C + (G_eff * H_eff) = C + C = 2C`。

## 2. Baseline / Advantage

沿用 MTPO 的 baseline 思路，但 STPO 把“选答案”和“训 confidence”拆开了：

### 2.1 Answer advantage（Round-1）
- Round-1 的 reward 来自 `answer_reward_funcs`（在 answer-only completion 上计算）。
- **Baseline（按 prompt 分组）**：对同一 prompt 下 `C` 个 candidates 的 answer score 做 mean/std 得到 baseline，得到 `A_ans`。
- **作用位置**：`A_ans` 只作用在 Round-1 的 answer-only 样本的 answer 段 token 上。
  - 额外注意：answer-only 样本的 terminal `eos` 不会被纳入 answer loss（避免强化“`</answer>` 后立刻 EOS”导致 Round-2 续写崩溃）。

### 2.2 Confidence advantage（Round-2）
- Round-2 的 reward 来自 `confidence_reward_funcs`，只在 **answer+confidence 的样本**上计算。
- **Baseline（按 answer 分组）**：对每个 selected answer 下 `H_eff` 个 confidence 的 reward 做 mean/std 得到 `A_conf`。

Token mask：
- `A_ans` 只作用在 Round 1 的 answer-only 样本的 answer 段 token（共 `C` 个样本）。
- `A_conf` 只作用在 Round 2 的 answer+confidence 样本的 confidence 段 token（共 `C` 个样本）。

## 3. 关键参数

在 `STPOConfig`（训练参数）里：
- `num_answer_candidates`：Round 1 采样的 candidate answers 数 `C`
- `num_answer_selected`：每个 prompt 期望选出的 answers 数 `G_target`（实际会变成 `G_eff`）
- `max_answer_length` / `max_confidence_length`
- `answer_stop_str` / `confidence_stop_str`
- `apply_answer_loss_on_first_confidence_only`：当前 STPO 已改为 answer-only + confidence 两批样本更新，保留该参数仅为兼容（不再影响 STPO 的 answer loss）。

在 `STPOScriptArguments`（脚本参数）里：
- `answer_selection_reward_funcs`：用于选 answer 的 reward 函数（在 answer-only completion 上计算）
- `answer_selection_reward_weights`：选 answer 的 reward 权重（可选）
- `answer_selection_format_pattern`：选 answer 时的格式 pattern（默认建议 `ta`）
- `answer_selection_strategy`：选 answer 的策略（默认 `random`；可选 `topk` / `balanced_accuracy`）
- `answer_selection_balanced_correct_fraction`：`balanced_accuracy` 时正确样本占比（默认 0.5）
- `answer_reward_funcs` / `answer_reward_weights`：Round-1 的训练 reward（只更新 answer 段 token；若不配置默认等于 `answer_selection_reward_funcs`）
- `confidence_reward_funcs` / `confidence_reward_weights`：Round-2 的训练 reward（只更新 confidence 段 token；若不配置默认等于旧字段 `reward_funcs`）
- `metric_reward_funcs`：只用于日志的 reward（不参与训练，用来记录 mean confidence 等指标）

（兼容）旧字段：
- `reward_funcs` / `reward_weights`：未配置 `confidence_reward_funcs` 时仍可用；建议迁移到 `confidence_reward_funcs` / `confidence_reward_weights`。

## 4. Reward funcs 如何“加在不同部分上”

配置里有 4 组 reward（前三组影响训练，最后一组只打点）：

### 4.1 `answer_selection_reward_funcs`（只用于 selection，不产生梯度）
- 这些 reward 在 **answer-only completion** 上计算，用来给 `C` 个 candidates 打分并选择进入 Round-2 的 answers。
- **强烈建议包含** `format_answer_segment`：它既用于 selection 的 score，也用于“format-gated selection”的 eligibility（只选 format 正确的 answers）。

### 4.2 `answer_reward_funcs`（Round-1：只更新 answer token）
- 这些 reward 在 **answer-only completion** 上计算，形成 `A_ans`，并且只乘 `answer_token_mask`（只更新 Round-1 的 answer 段 token）。
- 典型配置：`[format_answer_segment, accuracy]`（同时保证格式与正确性）。

### 4.3 `confidence_reward_funcs`（Round-2：只更新 confidence token）
- 这些 reward 在 **answer+confidence completion** 上计算，形成 `A_conf`，并且只乘 `confidence_token_mask`（只更新 Round-2 的 confidence 段 token）。
- 若 selected answer 的 answer 段 format 不正确，则对应的 Round-2 样本会被标记为 invalid，**这些 reward 会被乘 0（不计入训练）**。
- 建议把它保持“纯 confidence”：例如只用 `[format_confidence_segment, brier]` 来训练校准与格式。

### 4.4 `metric_reward_funcs`（仅日志，不参与训练）
- 这些 reward 在 Round-2 样本上计算，仅用于记录指标（例如 `stpo/mean_confidence`），不会进入 loss。
- 用它来替代“把 `reward_weights` 设为 0.0 只为了打点”的做法。

## 5. 训练/日志指标（常见）

STPO 训练时常看的指标含义（wandb key）：

- `stpo/format_answer_close_ratio`：Round-1 candidates 里能找到闭合 `</answer>` 的比例（用于判断 answer 段是否容易崩）。
- `stpo/answer_finish_reason_stop_ratio`：Round-1 candidates 的 vLLM `finish_reason == stop` 比例（停止条件是否有效）。
- `stpo/format_answer_segment_ratio`：Round-1 candidates 里 answer 段 format 正确的比例（由 `format_answer_segment` 判定）。
- `stpo/format_answer_selected_ratio`：被选中的 answers 里 answer 段 format 正确的比例（按“被选中的 answer 个数”统计）。
- `stpo/num_answer_selected_eff`：平均每个 prompt 的 `G_eff`。
- `stpo/num_confidence_generations_eff`：平均每个 prompt 的 `H_eff`。
- `valid_answer_ratio`：Round-2 confidence 样本里“其对应 answer 段 format 正确”的比例（按“confidence 样本数”统计，会被 `H_eff` 加权）。
- `stpo/format_confidence_segment_ratio`：Round-2 样本中 confidence 段 format 正确的比例（不按 answer validity mask）。
- `stpo/format_confidence_segment_ratio_valid_answer`：同上，但只统计 answer 段有效的 Round-2 样本。
- `stpo/mean_confidence` / `stpo/mean_confidence_valid_answer`：Round-2 的平均 confidence（以及仅对有效 answer 的平均）。
- `stpo/confidence_one_or_zero_ratio` / `stpo/confidence_one_or_zero_ratio_valid_answer`：Round-2 confidence 接近 0 或 1 的比例（以及仅对有效 answer 的比例）。
- （注）`stpo/format_confidence_segment_ratio` / `stpo/mean_confidence` / `stpo/confidence_one_or_zero_ratio` 只有在 `confidence_reward_funcs` 或 `metric_reward_funcs` 里配置了对应 reward 时才会记录。
- `rewards/<name>`：每个 reward 函数的均值（STPO 会分别在 Round-1 / Round-2 上计算对应的 reward，并统一用该命名风格记录）。
- `rewards/<name>/group_std`：同一 prompt 的 `C` 个 Round-2 样本在该 reward 上的组内方差水平（越大表示不同 confidence rollout 差异越大）。
- `rewards/<name>/answer_std`：同一 answer 的 `H_eff` 个 confidence rollout 在该 reward 上的方差水平（均值后再对 prompt 平均）。
- `spans/answer_length` / `spans/confidence_length` / `spans/split_ratio`：生成长度相关统计（只统计有效样本；candidate 样本不参与 confidence-length）。

## 6. Reward 函数释义（当前实现）

这些 reward 的实现位于 `src/RLCR/reward_fns.py`：

- `format_answer_segment`：只检查 “`<think>...</think><answer>...</answer>`” 片段是否格式正确（按最后一个 `</answer>` 切分）。
- `format_confidence_segment`：只检查 answer 之后的片段是否形如 “`<analysis>...</analysis><confidence>...</confidence>`”，并检查最后一个 confidence 能解析为 `[0,1]` 的 float。
  - 当 `format_pattern` 为 `tabc`（analysis 在 answer 之后）时：`<analysis>...</analysis>` **必须出现**（否则为 0）。
  - 其它 `format_pattern` 下为兼容历史实现：`<analysis>` 仍允许缺省。
- `format`：检查完整输出的标签顺序是否符合 `format_pattern`（例如 `tabc`），并对 confidence 数值做范围校验。
- `accuracy`（STPO 中实际使用 `accuracy_answer_segment`）：只 gate answer 段格式（`<think>...</think><answer>...</answer>`），不依赖 confidence 段格式，再用 `math_verify` 判断正确性。
- `brier`（当启用 `format_confidence_segment` 时 STPO 中实际使用 `brier_confidence_segment`）：correctness 来自 answer 段，brier 只 gate confidence 段格式，避免被 full-format gate 绑死。
- `log_likelihood`：在满足完整 `format_pattern` 的前提下，计算 `log p(y|confidence)`（格式不合法给一个很小的负值）。
- `mean_confidence`：抽取最后一个 `<confidence>` 的数值并 clip 到 `[0,1]`（不做完整格式校验，属于弱正则）。
- `confidence_one_or_zero`：鼓励 confidence 接近 0 或 1（不做完整格式校验，属于弱正则）。

## 7. 如何加速 rollout（不改代码）

Rollout 的主要成本来自“生成的 token 总数”和“vLLM 的吞吐/批处理效率”。常用加速手段（按收益从大到小）：

1) **减少生成 token 总量**
   - 降低 `num_answer_candidates (C)`：生成开销近似线性下降（但会牺牲探索/选择空间）。
   - 降低 `max_answer_length` / `max_confidence_length`：直接减少 token。
   - 确保 `answer_stop_str` / `confidence_stop_str` 能稳定触发：否则会经常跑满 `max_tokens`，速度会显著变慢。

2) **提高 vLLM 吞吐**
   - 提高 `vllm_gpu_memory_utilization`（在不 OOM 的前提下）以便 vLLM 更高并发/更少 swap。
   - 合理增大 `generation_batch_size` / `steps_per_generation`：让 vLLM 一次 generate 吃到更大的 batch（通常吞吐更好）。
   - 如果硬件允许，增大 `vllm_tensor_parallel_size`，或使用 `vllm_mode=server` 把 vLLM 放到单独 GPU/节点，减少与训练抢资源。

3) **降低 reward 计算开销（间接提升整体吞吐）**
   - `accuracy/brier/log_likelihood` 会触发解析与校验（`math_verify`），有时会成为 CPU 侧瓶颈；减少 reward 个数/减少调用频率可以明显加速“整体 wall time”，即使生成速度不变。

（注）当前实现会按 `H_eff` 分桶多次调用 vLLM `generate`；对 `C=32` 这种设置，`H_eff` 只会落在 32 的少数因子集合里（最多几个桶），通常不是主要开销来源。

## 8. 文件入口

- 训练入口：`src/RLCR/STPO_runner.py`
- Trainer：`src/RLCR/STPO_Trainer.py`
- 示例配置：`src/RLCR/configs/Qwen2_5-3B-Instruct/math/STPO.yaml`

## 9. 启动命令（示例）

`accelerate launch --num_processes 4 --config_file deepspeed.yaml STPO_runner.py --config configs/Qwen2_5-3B-Instruct/math/STPO.yaml`
