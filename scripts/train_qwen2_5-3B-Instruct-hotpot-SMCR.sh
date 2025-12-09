export CUDA_VISIBLE_DEVICES=1,2,3,4
export WANDB_ENTITY=rl_confidence
mkdir -p log
nohup accelerate launch --num_processes 4 --config_file deepspeed.yaml SMCR_runner.py \
    --config configs/Qwen2_5-3B-Instruct/hotpot/SMCR.yaml > log/Qwen2_5-3B-Instruct-hotpot-SMCR.log 2>&1 &
