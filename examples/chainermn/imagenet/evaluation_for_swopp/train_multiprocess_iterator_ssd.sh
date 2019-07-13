#!/bin/bash

if [ $# -ne 2 ]; then
    echo 'Usage: sh this_script {np} {timestamp}'
    exit 1
fi

echo 'This is train_multiprocess_iterator_ssd.sh'

np=${1}
CURRENT_DATETIME=${2}

export LD_LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

export PATH=/system/apps/cuda/10.1/bin:$PATH
export CPATH=/system/apps/cudnn/7.5.0/cuda10.1/include:/system/apps/cuda/10.1/include:$CPATH
export LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:$LIBRARY_PATH

export CUDA_HOME=/system/apps/cuda/10.1
export CUDA_PATH=/system/apps/cuda/10.1

ROOT="/work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp"
OUT=${ROOT}/results/multiprocess_iterator_ssd/${np}/${CURRENT_DATETIME}

LOG_DIR=${ROOT}/logs/multiprocess_iterator_ssd/${np}/${CURRENT_DATETIME}
LOG_STDERR=${LOG_DIR}/stderr_log

mkdir -p $OUT
mkdir -p $LOG_DIR

DUMMY_FILE="/tmp/dummy.${CURRENT_DATETIME}.`hostname`"
LOCK_FILE="/tmp/lock_file.${CURRENT_DATETIME}.`hostname`"
MPSTAT_LOG_FILE="/${LOG_DIR}/mpstat_`hostname`"
MPSTAT_PROC=-1

touch $DUMMY_FILE

# Copying imagenet files with ONLY ONE PROCESS for each node
copy_start_time=`date +%s`
if ! ln -s $DUMMY_FILE $LOCK_FILE; then
    while [ -e $LOCK_FILE ]; do
      sleep 1
    done
else
    echo `hostname`': got the lock'
    echo `hostname`': start copying the dataset'

    echo `hostname`': s: cp /work/NBB/serihiro/dataset/imagenet/256x256_all.tar /scr'
    cp -pr /work/NBB/serihiro/dataset/imagenet/256x256_all.tar /scr
    echo `hostname`': f: cp /work/NBB/serihiro/dataset/imagenet/256x256_all.tar /scr'
    
    cd /scr
    
    echo `hostname`': s: tar -xf 256x256_all.tar'
    tar -xf 256x256_all.tar
    echo `hostname`': f: tar -xf 256x256_all.tar'
    
    cd /scr/256x256_all

    echo `hostname`': s: python2 labeling.py train val'
    python2 labeling.py train val
    echo `hostname`': f: python2 labeling.py train val'

    echo `hostname`': finish copying the dataset'
    rm -rf $LOCK_FILE
    echo `hostname`': released the lock'

    
    echo `hostname`": start mpstat -P ALL 1 > $MPSTAT_LOG_FILE &"
    mpstat -P ALL 1 > $MPSTAT_LOG_FILE &
    MPSTAT_PROC=$!
    echo `hostname`": finish mpstat -P ALL 1 > $MPSTAT_LOG_FILE &"

    /usr/sbin/dropcaches 3
fi
copy_end_time=`date +%s`
time=$((copy_end_time - copy_start_time))
echo "`hostname`: copy_elapsed_time: ${time}"

/work/1/NBB/serihiro/venv/default/bin/python ${ROOT}/scripts/train_imagenet_extended.py \
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
  --epoch 1 \
  --out ${OUT} \
  --communicator pure_nccl 2>> ${LOG_STDERR}
  # --communicator hierarchical 2>> ${LOG_STDERR}
  # --communicator pure_nccl 2>> ${LOG_STDERR}


if [ $MPSTAT_PROC -ne -1 ]; then
    echo `hostname`": start kill -SIGINT MPSTAT_PROC"
    kill -INT $MPSTAT_PROC
    echo `hostname`": finish kill -SIGINT MPSTAT_PROC"
fi

