#!/usr/bin/env bash
set -euo pipefail

# 打包 wandb/offline-run-*，如果 run 后续有新文件或内容变更，会再次打包。
# 环境变量可覆盖:
#   ROOT_DIR=/path/to/wandb SNAP_DIR=/path/to/wandb_snap CHUNK_SIZE=10485760 ./wandb_snapshot.sh

ROOT_DIR=${ROOT_DIR:-$(realpath /tmp/wandb)}
SNAP_DIR=${SNAP_DIR:-$(realpath ./wandb_snap)}
CHUNK_SIZE=${CHUNK_SIZE:-10485760}  # 10MB 默认分块
TMP_DIR="${SNAP_DIR}/tmp_$(date +%s)"
LAST_FILE="${SNAP_DIR}/.last_sync"

mkdir -p "$SNAP_DIR"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

run_changed() {
  local run="$1"
  # 首次运行，全量
  [[ ! -f "$LAST_FILE" ]] && return 0

  # 目录本身是新建的
  [[ "$run" -nt "$LAST_FILE" ]] && return 0

  # 目录下有文件比上次新
  if find "$run" -type f -newer "$LAST_FILE" -print -quit | read -r _; then
    return 0
  fi

  # 对应的 run.wandb 有更新
  [[ -f "$run.wandb" && "$run.wandb" -nt "$LAST_FILE" ]] && return 0

  return 1
}

collect_runs() {
  while IFS= read -r dir; do
    run_changed "$dir" && printf '%s\n' "$dir"
  done < <(find "$ROOT_DIR" -maxdepth 1 -type d -name 'offline-run-*' | LC_ALL=C sort)
}

mapfile -t RUNS < <(collect_runs)
[[ ${#RUNS[@]} -eq 0 ]] && { echo "Nothing to package (no new/updated offline-run-* since last sync)."; exit 0; }

echo ">>> 本次同步的目录："
printf '%s\n' "${RUNS[@]}"

# 复制需要的目录
for run in "${RUNS[@]}"; do
  base=$(basename "$run")
  cp -a "$run" "$TMP_DIR/"
  [[ -f "$run.wandb" ]] && cp -a "$run.wandb" "$TMP_DIR/"
done

# 冻结时间戳，避免 tar 告警
find "$TMP_DIR" -exec touch {} +

# 分块压缩
ARCHIVE_PREFIX="$SNAP_DIR/wandb_$(date +%F-%H%M%S)"
echo ">>> 开始分块压缩，每块 ≤ $(( CHUNK_SIZE / 1024 / 1024 )) MB ..."
tar -C "$TMP_DIR" -czf - . 2>/dev/null | \
    split -b "$CHUNK_SIZE" -d -a 3 - "$ARCHIVE_PREFIX.part"

# 更新时间戳并清理
touch "$LAST_FILE"

echo ">>> 完成！分块文件保存在 $SNAP_DIR"
ls -lh "$ARCHIVE_PREFIX".part*
