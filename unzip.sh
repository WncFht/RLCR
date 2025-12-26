#!/usr/bin/env bash
set -euo pipefail

# RLCR 项目根目录
BASE_DIR="/home/fanghaotian-20250830/src/RLCR"
ARCHIVE_NAME="RLCR.tar.gz"
ARCHIVE_PATH="${BASE_DIR}/${ARCHIVE_NAME}"

# 需要清理 / 解压出的目录
DIR_RESULTS="${BASE_DIR}/results"
DIR_CONFIGS="${BASE_DIR}/configs"
DIR_SCRIPTS="${BASE_DIR}/scripts"
DIR_EVAL_CONFIGS="${BASE_DIR}/eval_configs"

echo "目标压缩包：${ARCHIVE_PATH}"

if [ ! -f "${ARCHIVE_PATH}" ]; then
    echo "错误：压缩包不存在：${ARCHIVE_PATH}" >&2
    exit 1
fi

echo "删除已有目录（如果存在）..."
rm -rf "${DIR_RESULTS}" "${DIR_CONFIGS}" "${DIR_SCRIPTS}" "${DIR_EVAL_CONFIGS}"

echo "开始解压..."
tar -xzf "${ARCHIVE_PATH}" -C "${BASE_DIR}"

echo "解压完成。当前目录结构："
ls -d "${DIR_RESULTS}" "${DIR_CONFIGS}" "${DIR_SCRIPTS}" "${DIR_EVAL_CONFIGS}"

echo "解压路径：${BASE_DIR}"
