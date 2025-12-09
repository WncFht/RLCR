export CUDA_VISIBLE_DEVICES=1,2,3,4
export WANDB_ENTITY=rl_confidence
mkdir -p log
nohup accelerate launch --num_processes 4 --config_file deepspeed.yaml rl_runner.py \
    --config configs/Qwen2_5-3B-Instruct/math/RLCR.yaml > log/Qwen2_5-3B-Instruct-math-RLCR.log 2>&1 &
