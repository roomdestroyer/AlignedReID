#!/bin/bash
# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
set -e

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh [MODEL_PATH] [IMAGE_PATH]"
echo "=============================================================================================================="

export ASCEND_HOME=/usr/local/Ascend
export ARCH_PATTERN=x86_64-linux
export ASCEND_VERSION=nnrt/latest


rm -rf dist
mkdir dist
cd dist
cmake ..
make -j
make install
./AlignedReID