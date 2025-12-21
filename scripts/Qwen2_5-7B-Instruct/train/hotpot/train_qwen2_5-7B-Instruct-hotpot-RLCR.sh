source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate rlcr
export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/rlcr/lib:$LD_LIBRARY_PATH
export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/rlcr/bin:$PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

export HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
export GREC_DIR=$HOME_DIR/GRec
export WANDB_DIR=$GREC_DIR
export WANDB_LOG_MODEL=false
export WANDB_ENTITY=rl_confidence

export HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
export RLCR_DIR=$HOME_DIR/RLCR

cd $RLCR_DIR

mkdir -p log
nohup accelerate launch \
    --num_processes 4 \
    --config_file deepspeed.yaml rl_runner.py \
    --config configs/Qwen2_5-7B-Instruct/hotpot/RLCR.yaml \
    > >(tee log/Qwen2_5-7B-Instruct-hotpot-RLCR.log) 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "To stop training: kill $PID"
wait "$PID"