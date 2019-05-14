#!/bin/bash

module load cuda/9.1.85
export LD_LIBRARY_PATH=/home/serizawa/lib/cudnn/lib64:$LD_LIBRARY_PATH
source /home/serizawa/python36_test/prefetch_iterator/chainer/venv/bin/activate

ROOT="/home/serizawa/python36_test/prefetch_iterator/chainer/examples/imagenet"
CONFIG_BASE="${ROOT}/resnet50_config"
TRAIN_SCRIPT="${ROOT}/train_imagenet_extended_100times.py"

python ${TRAIN_SCRIPT} `config2args "${CONFIG_BASE}/multiprocess_iterator/32_1000_16.json"`

