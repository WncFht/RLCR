export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_ENTITY=rl_confidence
nohup accelerate launch --num_processes 4 --config_file deepspeed.yaml rl_runner.py \
    --config configs/Qwen2_5-3B-Instruct/hotpot/RLCR-mo.yaml > Qwen2_5-3B-Instruct-hotpot-RLCR-mo.log 2>&1 &
