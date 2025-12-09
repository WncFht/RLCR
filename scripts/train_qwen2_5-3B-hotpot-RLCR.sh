export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p log
nohup accelerate launch --num_processes 4 --config_file deepspeed.yaml rl_runner.py \
    --config configs/Qwen2_5-3B/hotpot/RLCR.yaml > log/Qwen2_5-3B-hotpot-RLCR.log 2>&1 &
