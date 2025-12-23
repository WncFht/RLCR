source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate rlcr
export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/rlcr/lib:$LD_LIBRARY_PATH
export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/rlcr/bin:$PATH

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

export HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
export GREC_DIR=$HOME_DIR/GRec
export WANDB_DIR=$GREC_DIR
export WANDB_LOG_MODEL=false
export WANDB_ENTITY=rl_confidence

export HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
export RLCR_DIR=$HOME_DIR/RLCR

cd $RLCR_DIR

#HOTPOTQA Models
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/commonsenseqa.json
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/gpqa.json
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/gsm8k.json
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/hotpot-eval-em.json
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/hotpot-vanilla-eval-em.json
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/math-500.json
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/simpleqa.json
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/trivia.json
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/big-math-digits.json

# #MATH Models 
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Math-models/commonsenseqa.json
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Math-models/gpqa.json
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Math-models/gsm8k.json
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Math-models/hotpot-vanilla-eval-em.json
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Math-models/math-500.json
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Math-models/simpleqa.json
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Math-models/trivia.json


