## HOTPOT (4 GPU config) 
# RLVR
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file deepspeed.yaml rl_runner.py --config configs/Qwen-7B/hotpot/RLVR.yaml
# RLCR
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file deepspeed.yaml rl_runner.py --config configs/Qwen-7B/hotpot/RLCR.yaml


## MATH (6 GPU config) 
# RLVR
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes 6 --config_file deepspeed.yaml rl_runner.py --config configs/Qwen-7B/math/RLVR.yaml
# RLCR
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes 6 --config_file deepspeed.yaml rl_runner.py --config configs/Qwen-7B/math/RLCR.yaml
# SFT+RLCR
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes 6 --config_file deepspeed.yaml rl_runner.py --config configs/Qwen-7B/math/RLCR-SFT.yaml

## The generation batch size = num_processes * per_device_train_batch_size * gradient_accumulation_steps
## If more gpus are used, training can be sped up by reducing the gradient accumulation steps and increasing num_processes
## For 7B model, generally a minimum of 4 gpus is needed for training  





