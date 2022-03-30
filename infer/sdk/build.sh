#!/bin/bash

# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.

set -e

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash build.sh [mode]      mode = [ infer | load_eval | infer_eval ]  "
echo "for example: bash build.sh infer"
echo "=============================================================================================================="

# Simple log helper functions

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

# to set PYTHONPATH, import the StreamManagerApi.py
export PYTHONPATH=$PYTHONPATH:${MX_SDK_HOME}/python

mode=$1

python3.7 main.py \
--mode "${mode}" \

exit 0