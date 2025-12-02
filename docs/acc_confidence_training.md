# acc+confidence 双目标 RLCR 训练方案（多轮 Rollout 版本）

## 目标概述
- 解决“同一 prompt 的 confidence 奖励依赖不同 answer”带来的偏差：每条 confidence 必须针对同一个 answer rollout 出来，而不是和其他 answer 混合归一化。
- 采用多轮对话式 RL：第一轮采样 answer，第二轮在固定 answer 的上下文中采样 confidence。两个轮次的 tokens、reward、advantage 全部解耦，但仍共享一次完整的 RL 更新。
- 可独立控制 answer 与 confidence 的抽样数量（`num_generations_answer`、`num_generations_confidence`），以平衡探索与算力开销。

## 奖励设计
### Accuracy reward（第 1 轮）
第一轮生成 `think+answer`。对第 \(j\) 个 prompt 的第 \(k\) 次 answer（记作 \(a_{j,k}\)）计算 `verify`：
\[
\text{acc\_reward}_{j,k} = \begin{cases}
1 & \text{若 } \text{verify}(a_j, a_{j,k}) = \text{True},\\
0 & \text{否则}.
\end{cases}
\]
沿用 `reward_fns.accuracy_reward`。

### Confidence reward（第 2 轮，Brier）
第二轮生成 `confidence`，但上下文固定为“原 prompt + 第 1 轮生成出的 answer”。设第 \(k\) 个 answer 的第 \(h\) 次 confidence 预测为 \(c_{j,k,h} \in [0,1]\)。对应的准确性标签仍然是 \(\text{acc\_reward}_{j,k}\)。Brier 奖励：
\[
\text{conf\_reward}_{j,k,h} = 1 - (\text{acc\_reward}_{j,k} - c_{j,k,h})^2.
\]
若格式校验或解析失败，奖励置 0。

## 多轮 Rollout 设计
### 对话模板
1. **轮次一（Answer Generation）**
   - 输入：系统提示 + 用户 prompt。
   - 模型输出：`<think>…</think><answer>…</answer>`，可选 `<analysis>`。
   - 采样 `num_generations_answer = G_a` 条 answer。
2. **轮次二（Confidence Query）**
   - 对于第 \(k\) 条 answer，将它追加到对话历史作为 assistant 回复。
   - 向模型追加一条新的用户消息，例如：“请基于上面的回答给出 0-1 之间的置信度（只输出 `<confidence>`）。”
   - 在这个上下文中采样 `num_generations_confidence = G_c` 条 confidence。
   - 由于上下文包含完整 answer，所有 \(c_{j,k,h}\) 都严格对应同一个 \(a_{j,k}\)。

该策略形成树状对话（prompt → answer → confidence）。数据组织上需要记录：
- `answer_ids[j,k,t]`：第 \(j\) 个 prompt、第 \(k\) 条 answer 的 token 序列；
- `confidence_ids[j,k,h,t]`：对应 confidence 的 token 序列；
- `answer_span_mask`、`confidence_span_mask`：分别在全局序列里标记 token 位置（见下一节）。

### Generation 数量与批组织
- **第一轮** `G_a` 决定了传统 GRPO 的组大小，仍按 prompt 分组做均值/方差归一化。
- **第二轮** 每条 answer 拓展出 `G_c` 个 confidence。可以视为 `G_a × G_c` 个兄弟节点，但归一化时应先在每个 answer 内求均值，再在 prompt 级别聚合，从而避免把不同答案的 confidence 混为一谈。
- 在实现上可以复用现有的 `num_generations` 配置，但推荐新增两个参数并在 Trainer 中明确：第一轮在 vLLM 里一次生成 `G_a` 条；第二轮针对每个 answer 单独调用生成，或把 `(prompt, answer)` 二元组批量化到下一次 vLLM 请求。

## 段落 mask 与 token 对齐
整体思路：**按回合划分可学习 token**。第 1 轮把 `<think> + <answer>` 这段全部开放给 answer reward；第 2 轮把 `<analysis>（若有） + <confidence>` 这段开放给 confidence reward。mask 只需要两个集合：

- `M^{\text{ans}}_{j,k}`：第 \(j\) 个 prompt、第 \(k\) 条 answer 的 think+answer 区域。定位方法：
  1. 解析 `<think>`、`</think>`、`<answer>`、`</answer>` 标签；
  2. 取 `think` 起点到 `answer` 结束为整个 mask（如果没有 `<analysis>`，就以 `<confidence>` 或 `<eos>` 作为终点）。
  3. 若中途包含 `<analysis>`，但它只在第二轮出现，这里不需要关注。

- `M^{\text{conf}}_{j,k,h}`：第 2 轮生成结果里，`<analysis>`（如存在）+`<confidence>` 的 token 区域。具体：
  1. 若模型在回合二输出 `<analysis>`，则从其起始到 `<confidence>` 结束全部记为 mask；
  2. 如果没有 `<analysis>`，则直接从 `<confidence>` 起始到结束；
  3. 其余 token（比如 system/user 追加的提示）都被 mask=0。

实现注意：
- 两个回合完全分开采样，token 序列互不重叠，因此把它们堆叠进同一 batch 后，只需要记录 `sample_type ∈ {answer, confidence}` 以及对应 mask，训练时按类型选取即可。
- **不要**把 mask 限定在 `<answer>` 或 `<confidence>` 内部，否则 think/analysis 的 token 无法得到 reward 信息，难以学到“如何推理”或“如何解释置信度”。

## 多 reward 的优势计算与 loss 设计
多轮 rollout 后，answer 与 confidence 的 reward 分布不同，需要分开归一化，再映射到对应 tokens。

### Answer 优势（think+answer）
对 prompt \(j\) 的 \(G_a\) 个 answer：
\[
\mu^{\text{ans}}_j = \frac{1}{G_a}\sum_{k=1}^{G_a} \text{acc\_reward}_{j,k}, \quad
\sigma^{\text{ans}}_j = \sqrt{\frac{1}{G_a}\sum_{k=1}^{G_a} (\text{acc\_reward}_{j,k} - \mu^{\text{ans}}_j)^2}.
\]
\[
A^{\text{ans}}_{j,k} = \frac{\text{acc\_reward}_{j,k} - \mu^{\text{ans}}_j}{\sigma^{\text{ans}}_j + \varepsilon}.
\]
若关闭归一化则直接减均值。投射到 token 维度：
\[
\mathcal{A}^{\text{ans}}_{j,k,t} = \lambda_{\text{ans}} \cdot A^{\text{ans}}_{j,k} \cdot M^{\text{ans}}_{j,k,t}.
\]

### Confidence 优势（analysis+confidence）
每条 answer 再派生 \(G_c\) 个 confidence，推荐分两级归一化：
1. **同一 answer 内**：  
\[
\mu^{\text{conf}}_{j,k} = \frac{1}{G_c}\sum_{h=1}^{G_c} \text{conf\_reward}_{j,k,h}, \quad
\sigma^{\text{conf}}_{j,k} = \sqrt{\frac{1}{G_c}\sum_{h=1}^{G_c} (\text{conf\_reward}_{j,k,h} - \mu^{\text{conf}}_{j,k})^2}.
\]
2. **再在 prompt 级别**：为了避免不同 answer 的 confidence 互相影响，可再次计算 prompt 级统计或直接使用上一步结果。

最终投射：
\[
\mathcal{A}^{\text{conf}}_{j,k,h,t} = \lambda_{\text{conf}} \cdot \frac{\text{conf\_reward}_{j,k,h} - \mu^{\text{conf}}_{j,k}}{\sigma^{\text{conf}}_{j,k} + \varepsilon} \cdot M^{\text{conf}}_{j,k,h,t}.
\]
### PPO 损失
answer 与 confidence 样本共享 PPO/GRPO 框架，但 `loss_mask` 和 `advantages` 使用不同来源：
- Answer 样本：`loss_mask = M^{\text{ans}}`（think+answer），`advantages = \mathcal{A}^{\text{ans}}`；
- Confidence 样本：`loss_mask = M^{\text{conf}}`（analysis+confidence），`advantages = \mathcal{A}^{\text{conf}}`。

设 \(\ell_{i,t}\) 为标准 GRPO 双裁剪 token 损失，则
\[
\mathcal{L} = \frac{\sum_{(i,t)} \ell_{i,t} \cdot M^{\text{target}}_{i,t}}{\sum_{(i,t)} M^{\text{target}}_{i,t} + \eta},
\]
即可兼容 BNPO/DR-GRPO 的不同归一化方式。

## 训练流程（高层）
1. **第一轮生成**：使用 vLLM 从每个 prompt 采样 `G_a` 条 answer，解析 `<answer>` span，计算 `acc_reward`。
2. **第二轮生成**：对每条 answer 构建新的对话上下文，请模型输出 confidence（可一次批量化 `G_c` 条），解析 `<confidence>` span。
3. **奖励计算**：  
   - answer reward：`accuracy_reward`；  
   - confidence reward：`brier_reward`（输入为同一 answer 的标签）；  
   - 记录 `answer_mask`、`confidence_mask`。
4. **优势归一化**：按上一节公式分别计算 answer / confidence 的优势，叠成统一 tensor。
5. **PPO/GRPO 更新**：  
   - 重新计算 logprob / ref logprob / value，使用 `loss_mask` 与 `advantages` 做 token 级 loss；  
   - 一个 batch 内既包含 answer 样本也包含 confidence 样本，梯度会自动聚合。
6. **日志**：分别记录 answer / confidence 的 reward mean/std、两轮长度、`G_a`/`G_c` 的有效性等。

## 对代码的改动建议
1. **配置**：新增 `num_generations_answer`, `num_generations_confidence`, `lambda_ans`, `lambda_conf` 等参数；控制是否启用多轮模式。
2. **Rollout 控制器**：在 `rl_runner` 或 Trainer 内，把第一轮 completion 缓存下来，再根据缓存构造第二轮提示；需要标注 `tool_call` 式的 `(prompt, answer)` 对话 ID 以便归组。
3. **数据结构**：扩充 batched tensor，支持“同一 prompt 多条 answer + 子 confidence 样本”结构；在 `inputs` 中加入 `sample_type` 或 `turn_id`。
4. **奖励模块**：`reward_fns.py` 中的 `brier_reward` 需要接受 `(answer_text, confidence_text)` 配对的列表；也可在 Trainer 里调用两次 reward 函数分别处理。
5. **Trainer**：  
   - `_generate_and_score_completions` 拆成两段，或维护 state 以区分当前 turn；  
   - `advantages` 从 `[B]` 扩展为 `[B, T]`，并按 sample_type 选择对应 reward；  
   - Update/loss 阶段根据 `sample_type` 选择 mask 与优势。
6. **日志与可视化**：新增 answer vs confidence 的 reward/长度直方图，确保多轮逻辑稳定。

通过以上方案，confidence 的 reward 与 advantage 仅在与其对应的 answer 上计算，多轮 rollout 保证“同 answer 多 confidence”这一约束，从而避免交叉污染和不公平归一化。***
