#!/usr/bin/env bash
set -euo pipefail

# 压缩包存放目录（RLCR 项目根目录）
BASE_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/RLCR"
ARCHIVE_NAME="RLCR.tar.gz"
ARCHIVE_PATH="${BASE_DIR}/${ARCHIVE_NAME}"

# 要压缩的目录
DIR_RESULTS="${BASE_DIR}/results"
DIR_CONFIGS="${BASE_DIR}/configs"
DIR_SCRIPTS="${BASE_DIR}/scripts"
DIR_EVAL_CONFIGS="${BASE_DIR}/eval_configs"

echo "目标压缩包：${ARCHIVE_PATH}"

# 如果已有压缩包，先删除
if [ -f "${ARCHIVE_PATH}" ]; then
    echo "已存在压缩包，删除：${ARCHIVE_PATH}"
    rm -f "${ARCHIVE_PATH}"
fi

# 创建压缩包
echo "开始压缩..."
tar -czf "${ARCHIVE_PATH}" \
    -C "${BASE_DIR}" \
    "$(basename "${DIR_RESULTS}")" \
    "$(basename "${DIR_CONFIGS}")" \
    "$(basename "${DIR_SCRIPTS}")" \
    "$(basename "${DIR_EVAL_CONFIGS}")"

echo "压缩完成：${ARCHIVE_PATH}"
