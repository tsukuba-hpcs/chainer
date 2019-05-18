#!/bin/bash

module load cuda/9.1.85
export LD_LIBRARY_PATH=/home/serizawa/lib/cudnn/lib64:$LD_LIBRARY_PATH
source /home/serizawa/python36_test/prefetch_iterator/chainer/venv/bin/activate

ROOT="/home/serizawa/python36_test/prefetch_iterator/chainer/examples/imagenet"
CONFIG_BASE="${ROOT}/resnet50_config"
TRAIN_SCRIPT="${ROOT}/train_imagenet_extended.py"

rm -rf /work/imagenet/local_storage_base/*
/ppxsvc/bin/ppx-drop-caches
python ${TRAIN_SCRIPT} `config2args "${CONFIG_BASE}/prefetch_multiprocess_iterator/32_1000_8_16.json"`

