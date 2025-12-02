# acc+confidence 双目标 RLCR（单轮对话方案）

## 核心想法
- 仍然让模型一次生成完整输出：`<think>…</think><answer>…</answer><analysis>…</analysis><confidence>…</confidence>`。
- 只 rollout 一次，避免额外的 vLLM 调用和缓存逻辑。
- `accuracy_reward` 负责优化 `<think> + <answer>` 这段 token；`confidence_reward` 只作用在 `<analysis> + <confidence>`。
- 为了公平地训练 confidence：按照 answer 的正确/错误分成两组，在各自组内归一化，然后分组广播到对应 token。

## Token mask 设计
1. **Answer mask \(M^{\text{ans}}\)**  
   - 覆盖 `<think>` 和 `<answer>`，从 `<think>` 起点到 `<answer>` 结束。
   - 若 `analysis` 和 `confidence` 与答案使用相同序列，mask 需要确保在 `<analysis>` 之前终止。

2. **Confidence mask \(M^{\text{conf}}\)**  
   - 覆盖 `<analysis>`（如存在）与 `<confidence>`，从 `<analysis>` 起点（若无则 `<confidence>` 起点）到 `<confidence>` 结束。
   - 这样 confidence_reward 只会影响解释+置信度部分，不会影响 answer token。

训练时构建一个 `sample_type` 向量：`answer` 样本使用 `M^{\text{ans}}`；`confidence` 样本使用 `M^{\text{conf}}`。因为所有 token 来自同一条序列，实际实现可直接保存两个 mask 并在 loss 中区分。

## 奖励与优势计算
### Accuracy reward（think+answer）
和现有 RLCR 完全一致：  
\[
\text{acc\_reward}_i = \mathbf{1}\{\text{verify}(a_i, \hat{a}_i)\}
\]
按 prompt 分组做均值/方差归一化，得到 `A^{ans}`，再乘到 \(M^{\text{ans}}\)。

### Confidence reward（analysis+confidence）
1. **解析 confidence 值**：`brier_reward` 中已经提取 `<confidence>` 文本并与 `acc_reward` 计算 \(1 - (\hat{c}-\text{acc})^2\)。
2. **按 correctness 分组**：把同一个 prompt 内的样本拆成两组：
   - **Correct 组**：`acc_reward=1`；
   - **Wrong 组**：`acc_reward=0`。
3. **组内归一化**：对每一组分别计算均值/方差或 rank-based 归一化：
\[
\mu^{\text{conf}}_{\text{correct}} = \frac{1}{N_c}\sum_{i\in \text{correct}} \text{conf\_reward}_i, \quad
A^{\text{conf}}_{\text{correct},i} = \frac{\text{conf\_reward}_i - \mu^{\text{conf}}_{\text{correct}}}{\sigma^{\text{conf}}_{\text{correct}} + \varepsilon}
\]
错误组同理。若某组只有一个样本，直接设置优势为 0。
4. **广播到 token**：  
\[
\mathcal{A}^{\text{conf}}_{i,t} = \lambda_{\text{conf}} \cdot A^{\text{conf}}_i \cdot M^{\text{conf}}_{i,t}
\]

这样 correct 组会被鼓励提高/降低 confidence 到更接近真实准确度，wrong 组则鼓励输出低置信度。

## 训练流程
1. **Rollout**：保持现状（单轮 vLLM 生成），缓存 prompts/completions/completion_mask。
2. **Span & mask 提取**：解析 `<think>`, `<answer>`, `<analysis>`, `<confidence>` 的位置，生成 `M^{\text{ans}}`, `M^{\text{conf}}`。
3. **奖励计算**：
   - `accuracy_reward` → think+answer；
   - `brier_reward` → analysis+confidence。
4. **优势划分**：
   - `A^{ans}`：标准 GRPO；
   - `A^{conf}`： correct vs wrong 分组归一化。
5. **Loss**：
   - Answer loss：`loss_mask = M^{\text{ans}}`，`advantages = A^{ans}`；
   - Confidence loss：`loss_mask = M^{\text{conf}}`，`advantages = A^{conf}`；
   - 在实现上，可将两种样本拼成一个 batch（例如先 answer 再 confidence），或在张量维度上拼接两个 mask。

## 配置建议
- 新增 `lambda_conf`（confidence reward 权重）；
- 是否按 correctness 分组可加开关，防止过拟合；
- 允许用户选择 `scale_confidence_rewards`（控制是否做组内标准化）。

## 优势
- **无需多轮生成**：对话仍是一轮，训练复杂度与当前 RLCR 接近；
- **目标解耦**：think+answer 与 analysis+confidence 拥有不同 reward/mask，避免互相干扰；
- **公平性**：confidence 的对错分组保证 reward 仅与“同样正确/错误的答案”比较，避免 reward 混淆；
- **兼容现有代码**：只需在 reward 计算后构建两个 mask 与优势，`PPO loss` 逻辑无需大改。

## 实现要点
1. 在 `_generate_and_score_completions` 中解析 span，生成 `M^{\text{ans}}`, `M^{\text{conf}}`；
2. `rewards_per_func` 输出中保留 acc/conf 两列，供后续分组归一化；
3. `inputs` 结构增加 `answer_mask`, `confidence_mask`，或通过 `inputs["sample_type"]` 区分；
4. `_compute_loss` 内根据 `sample_type` 选择相应 mask 与优势，再执行 GRPO/BNPO；
5. 记录 separate metrics：正确样本的置信度均值、错误样本的置信度均值、confidence reward std 等。

这样即可在一轮 rollout 内同时优化答案正确率与置信度校准。***
