export CUDA_VISIBLE_DEVICES=0,1,2,3 
export WANDB_TEMP_DIR=/home/fanghaotian-20250830/src/.wandb_temp
nohup accelerate launch --num_processes 4 --config_file deepspeed.yaml rl_runner.py \
    --config configs/Qwen2_5-3B/hotpot/RLVR.yaml > Qwen2_5-3B-hotpot-RLVR.log 2>&1 &
