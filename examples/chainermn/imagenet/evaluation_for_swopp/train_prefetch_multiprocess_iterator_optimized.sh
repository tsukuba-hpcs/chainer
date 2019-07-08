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
OUT=${ROOT}/results/prefetch_multiprocess_iterator/${np}/${CURRENT_DATETIME}
LOG_STDERR=${ROOT}/logs/prefetch_multiprocess_iterator/${np}/${CURRENT_DATETIME}

mkdir -p $OUT
mkdir -p ${ROOT}/logs/prefetch_multiprocess_iterator/${np}

mkdir -p /scr/local_storage_base

# m(__)m
ps aux | grep mpstat | grep -v grep
ps aux | grep mpstat | grep -v grep | awk '{ print "kill -9", $2 }' | sh

DUMMY_FILE="/tmp/dummy.${CURRENT_DATETIME}.`hostname`"
LOCK_FILE="/tmp/lock_file.${CURRENT_DATETIME}.`hostname`"
MPSTAT_LOG_FILE="${ROOT}/logs/prefetch_multiprocess_iterator/${np}/${CURRENT_DATETIME}_mpstat/`hostname`.log"
MPSTAT_PROC=-1

mkdir -p ${ROOT}/logs/prefetch_multiprocess_iterator/${np}/${CURRENT_DATETIME}_mpstat

touch $DUMMY_FILE

if ! ln -s $DUMMY_FILE $LOCK_FILE; then
    echo `hostname`": LOCKED"

    while [ -e $LOCK_FILE ]; do
      echo `hostname`': waiting...'
      sleep 1
    done
    echo `hostname`': detected the lock has been released'  
else
    echo `hostname`": start mpstat -P ALL 60 > $MPSTAT_LOG_FILE &"
    mpstat -P ALL 60 > $MPSTAT_LOG_FILE &
    MPSTAT_PROC=$!
    echo `hostname`": finish mpstat -P ALL 60 > $MPSTAT_LOG_FILE &"
    rm -rf $LOCK_FILE
    /usr/sbin/dropcaches 3
fi

/work/1/NBB/serihiro/venv/default/bin/python ${ROOT}/scripts/train_imagenet_extended_optimized.py \
  /work/NBB/serihiro/dataset/imagenet/256x256_all/train.ssv \
  /work/NBB/serihiro/dataset/imagenet/256x256_all/val.ssv \
  --mean /work/NBB/serihiro/dataset/imagenet/256x256_all/mean.npy \
  --root /work/NBB/serihiro/dataset/imagenet/256x256_all/train \
  --local_storage_base /scr/local_storage_base \
  --arch resnet50 \
  --n_prefetch 1000 \
  --iterator prefetch_multiprocess \
  --prefetchjob 1 \
  --loaderjob 1 \
  --batchsize 32 \
  --val_batchsize 32 \
  --epoch 1 \
  --out ${OUT} \
  --communicator pure_nccl 2>> ${LOG_STDERR}

if [ $MPSTAT_PROC -ne -1 ]; then
    echo `hostname`": start kill -SIGINT $MPSTAT_PROC"
    kill -INT $MPSTAT_PROC
    echo `hostname`": finish kill -SIGINT $MPSTAT_PROC"
fi

