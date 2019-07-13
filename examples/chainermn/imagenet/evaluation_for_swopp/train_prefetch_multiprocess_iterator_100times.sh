#!/bin/bash

if [ $# -ne 2 ]; then
    echo 'Usage: sh this_script {np} {timestamp}'
    exit 1
fi

np=${1}
CURRENT_DATETIME=${2}

export LD_LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

export PATH=/system/apps/cuda/10.1/bin:$PATH
export CPATH=/system/apps/cudnn/7.5.0/cuda10.1/include:/system/apps/cuda/10.1/include:$CPATH
export LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:$LIBRARY_PATH

export CUDA_HOME=/system/apps/cuda/10.1
export CUDA_PATH=/system/apps/cuda/10.1

ROOT="/work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp"
LOG_STDERR=${ROOT}/logs/prefetch_multiprocess_100times/${np}/${CURRENT_DATETIME}

mkdir -p ${ROOT}/logs/prefetch_multiprocess_100times/${np}

/usr/sbin/dropcaches 3

/system/apps/cuda/10.1/bin/nvprof \
    -o ${ROOT}/logs/prefetch_multiprocess_100times/${np}/${CURRENT_DATETIME}.%q{OMPI_COMM_WORLD_RANK}.nvvp \
    /work/1/NBB/serihiro/venv/default/bin/python ${ROOT}/scripts/train_imagenet_extended_100times.py \
    /work/NBB/serihiro/dataset/imagenet/256x256_all/train.ssv \
    /work/NBB/serihiro/dataset/imagenet/256x256_all/val.ssv \
    --mean /work/NBB/serihiro/dataset/imagenet/256x256_all/mean.npy \
    --root /work/NBB/serihiro/dataset/imagenet/256x256_all/train \
    --local_storage_base /scr/local_storage_base \
    --arch resnet50 \
    --n_prefetch 1000 \
    --iterator prefetch_multiprocess \
    --prefetchjob 2 \
    --loaderjob 2 \
    --batchsize 32 \
    --val_batchsize 32 \
    --epoch 10 \
    --communicator pure_nccl 2>> ${LOG_STDERR}

