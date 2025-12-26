#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0,1,2 ./eval_runs_all.sh [--gpus LIST] [--python BIN] [--no-env] [--suite PATH] [--datasets LIST]

Notes:
  - --gpus overrides CUDA_VISIBLE_DEVICES.
  - LIST is like: 0,1,2 (no spaces).
  - One worker per GPU; each worker runs tasks serially on its GPU.
  - Default is to read datasets from the suite config.
EOF
  exit 2
}

GPU_SPEC=""
PY_BIN="python"
DO_ENV=true
SUITE_CFG="eval_configs/suite.json"
DATASET_LIST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus|--gpu|--cuda-visible-devices)
      [[ $# -ge 2 ]] || usage
      GPU_SPEC="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || usage
      PY_BIN="$2"
      shift 2
      ;;
    --no-env)
      DO_ENV=false
      shift
      ;;
    --suite|--config)
      [[ $# -ge 2 ]] || usage
      SUITE_CFG="$2"
      shift 2
      ;;
    --datasets|--tasks)
      [[ $# -ge 2 ]] || usage
      DATASET_LIST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      ;;
  esac
done

if $DO_ENV; then
  # Keep your original environment setup here (edit paths if needed).
  # conda 的 activate/deactivate 脚本通常不兼容 `set -u`（nounset），这里临时关闭避免报
  # `CONDA_BACKUP_*: unbound variable` 之类的错误。
  set +u
  source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate rlcr
  set -u
  export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/rlcr/lib:${LD_LIBRARY_PATH:-}
  export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/rlcr/bin:$PATH

  export WANDB_MODE=offline

  export HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
  export GREC_DIR=$HOME_DIR/GRec
  export WANDB_DIR=$GREC_DIR
  export WANDB_LOG_MODEL=false
  export WANDB_ENTITY=rl_confidence

  export RLCR_DIR=$HOME_DIR/RLCR
  cd "$RLCR_DIR"
fi

TASKS=()
if [[ -n "$DATASET_LIST" ]]; then
  IFS=',' read -r -a TASKS <<<"$DATASET_LIST"
else
  # Read dataset ids from the suite config without importing heavy deps.
  mapfile -t TASKS < <("$PY_BIN" - <<PY
import json
with open("$SUITE_CFG","r") as f:
    cfg=json.load(f)
for d in cfg.get("datasets",[]):
    if d.get("id"):
        print(d["id"])
PY
  )
fi

[[ ${#TASKS[@]} -gt 0 ]] || { echo "No tasks selected." >&2; exit 2; }

# If user didn't provide anything, default to a single GPU: 0.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0"
fi

if [[ -z "$GPU_SPEC" ]]; then
  GPU_SPEC="${CUDA_VISIBLE_DEVICES:-}"
fi
GPU_SPEC="${GPU_SPEC// /}"
[[ -n "$GPU_SPEC" ]] || GPU_SPEC="0"

IFS=',' read -r -a GPUS <<<"$GPU_SPEC"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "Empty GPU list." >&2; exit 2; }

FIFO="$(mktemp -u "${TMPDIR:-/tmp}/rlcr_eval_queue.XXXXXX")"
mkfifo "$FIFO"

PIDS=()
cleanup() {
  rm -f "$FIFO" 2>/dev/null || true
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

(
  for cfg in "${TASKS[@]}"; do
    printf '%s\n' "$cfg"
  done
) >"$FIFO" &
WRITER_PID=$!

worker() {
  local gpu="$1"
  local any_fail=0
  local cfg

  while IFS= read -r cfg; do
    [[ -n "$cfg" ]] || continue
    echo ">>> [gpu=$gpu] START $cfg" >&2
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$PY_BIN" evaluation.py --config "$SUITE_CFG" --dataset "$cfg"
    st=$?
    set -e
    if (( st != 0 )); then
      echo ">>> [gpu=$gpu] FAIL  $cfg (exit=$st)" >&2
      any_fail=1
    else
      echo ">>> [gpu=$gpu] DONE  $cfg" >&2
    fi
  done <"$FIFO"

  return "$any_fail"
}

echo ">>> GPUs: ${GPUS[*]}" >&2
echo ">>> Tasks: ${#TASKS[@]}" >&2

for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
  PIDS+=("$!")
done

set +e
wait "$WRITER_PID"
rc=0
for pid in "${PIDS[@]}"; do
  wait "$pid"
  st=$?
  (( st != 0 )) && rc=1
done
set -e

exit "$rc"
