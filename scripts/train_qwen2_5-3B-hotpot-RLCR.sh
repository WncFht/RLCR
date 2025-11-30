export CUDA_VISIBLE_DEVICES=0,1,2,3 
nohup accelerate launch --num_processes 4 --config_file deepspeed.yaml rl_runner.py \
    --config configs/Qwen2_5-3B/hotpot/RLCR.yaml > Qwen2_5-3B-hotpot-RLCR.log 2>&1 &
