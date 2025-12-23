# MTPO（Multi-Turn Policy Optimization）训练说明

## 1. 核心思路（两轮 rollout）

MTPO 把一次采样拆成两轮：

1) **第 1 轮（Answer Rollout）**  
在同一个问题 `prompt` 下采样 `G = num_answer_generations` 个候选输出，只生成到 `</answer>` 为止：  
`<think>...</think><answer>...</answer>`

2) **第 2 轮（Confidence Rollout）**  
对第 1 轮得到的每个 `answer`，继续在其后采样 `H = num_confidence_generations` 次，只生成到 `</confidence>` 为止：  
`<analysis>...</analysis><confidence>...</confidence>`

最终一次“完整 completion”是第 1 轮与第 2 轮拼接得到的：
`<think>...</think><answer>...</answer><analysis>...</analysis><confidence>...</confidence>`

## 2. Baseline / Advantage 计算（你要的分组方式）

设每个 prompt 对应 `G` 个 answer，每个 answer 对应 `H` 个 confidence 采样，因此总采样数为：
`num_generations = G * H`

在 `src/RLCR/MTPO_Trainer.py` 中实现的 baseline 分组为：

- **Answer baseline：按 prompt 分组**  
对同一 prompt 下的 `G` 个 answer 的 *answer-reward* 做均值/方差，得到 answer 的优势 `A_ans`。  

- **Confidence baseline：按 answer 分组**  
对同一 answer 下的 `H` 个 confidence 的 *confidence-reward* 做均值/方差，得到 confidence 的优势 `A_conf`。

训练时用 token mask 把两部分优势分别施加到两段 token 上：
- `A_ans` 只作用在第 1 轮生成的 token（think+answer）
- `A_conf` 只作用在第 2 轮生成的 token（analysis+confidence）

默认 `apply_answer_loss_on_first_confidence_only=true`：每个 answer 的第 1 轮 token 只在该 answer 的第一个 confidence 样本上计算一次损失，避免第 1 轮梯度被 `H` 倍放大。

## 3. 可调参数（对应 `MTPOConfig`）

主要是下面这些（都在配置 YAML 里可写）：

- 采样结构：
  - `num_answer_generations`：第 1 轮每个 prompt 的 answer 数 `G`
  - `num_confidence_generations`：第 2 轮每个 answer 的 confidence 数 `H`
- 两轮长度与停止符：
  - `max_answer_length`：第 1 轮最大生成 token 数
  - `max_confidence_length`：第 2 轮最大生成 token 数
  - `answer_stop_str`：第 1 轮停止字符串（默认 `</answer>`）
  - `confidence_stop_str`：第 2 轮停止字符串（默认 `</confidence>`）
- 损失/归一化：
  - `apply_answer_loss_on_first_confidence_only`
  - `scale_rewards`、`beta`、`loss_type`、`epsilon/epsilon_high`、`num_iterations` 等（沿用 GRPO/BNPO 配置）
- 生成采样：
  - `temperature`
- 常规训练：
  - `learning_rate`、`per_device_train_batch_size`、`gradient_accumulation_steps`、`max_prompt_length` 等

## 4. 推荐 reward 组合

为了让“answer 段”和“confidence 段”分别有清晰的奖励信号，建议使用分段格式奖励：

- `format_answer_segment`：只检查 `<think>...</think><answer>...</answer>` 段
- `format_confidence_segment`：只检查 `<analysis>...</analysis><confidence>...</confidence>` 段

示例见：`src/RLCR/configs/Qwen2_5-3B-Instruct/hotpot/MTPO.yaml`

## 5. 启动命令

- 直接用 accelerate 启动：
  - `accelerate launch --num_processes 8 --config_file deepspeed.yaml MTPO_runner.py --config configs/Qwen2_5-3B-Instruct/hotpot/MTPO.yaml`

- 或者用脚本启动：
  - `bash src/RLCR/scripts/Qwen2_5-3B-Instruct/train/hotpot/train_qwen2_5-3B-Instruct-hotpot-MTPO.sh`
