#!/bin/bash

if [ $# -ne 1 ]; then
    echo 'Usage: sh this_script {np}'
    exit 1
fi

np=${1}

export LD_LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

export PATH=/system/apps/cuda/10.1/bin:$PATH
export CPATH=/system/apps/cudnn/7.5.0/cuda10.1/include:/system/apps/cuda/10.1/include:$CPATH
export LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:$LIBRARY_PATH

export CUDA_HOME=/system/apps/cuda/10.1
export CUDA_PATH=/system/apps/cuda/10.1

CURRENT_DATETIME=`date "+%Y%m%d_%H%M%S"`
ROOT="/work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp"
LOG_STDERR=${ROOT}/logs/multiprocess_iterator_ssd_10times/${np}/${CURRENT_DATETIME}

mkdir -p ${ROOT}/logs/multiprocess_iterator_ssd_10times/${np}

/system/apps/cuda/10.1/bin/nvprof \
    -o ${ROOT}/logs/multiprocess_iterator_ssd_10times/${np}/${CURRENT_DATETIME}.%q{OMPI_COMM_WORLD_RANK}.nvvp \
    /work/1/NBB/serihiro/venv/default/bin/python ${ROOT}/scripts/train_imagenet_extended_10times.py \
    /scr/256x256_all/train.ssv \
    /scr/256x256_all/val.ssv \
    --mean /scr/256x256_all/mean.npy \
    --root /scr/256x256_all/train \
    --local_storage_base /scr/local_storage_base \
    --arch resnet50 \
    --n_prefetch 1000 \
    --iterator multiprocess \
    --prefetchjob 2 \
    --loaderjob 2 \
    --batchsize 32 \
    --val_batchsize 32 \
    --epoch 10 \
    --communicator pure_nccl
    # --communicator pure_nccl 2>> ${LOG_STDERR}

