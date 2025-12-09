export CUDA_VISIBLE_DEVICES=1
export WANDB_ENTITY=debug
mkdir -p log
accelerate launch --num_processes 1 --config_file deepspeed.yaml SMCR_runner.py \
    --config configs/Qwen2_5-3B-Instruct/hotpot/SMCR.yaml > log/Qwen2_5-3B-Instruct-hotpot-SMCR-debug.log 2>&1
