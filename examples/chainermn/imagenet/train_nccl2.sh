#!/bin/bash

export LD_LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

export PATH=/system/apps/cuda/10.1/bin:$PATH
export CPATH=/system/apps/cudnn/7.5.0/cuda10.1/include:/system/apps/cuda/10.1/include:$CPATH
export LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:$LIBRARY_PATH

export CUDA_HOME=/system/apps/cuda/10.1
export CUDA_PATH=/system/apps/cuda/10.1

/work/NBB/serihiro/venv/default/bin/python /work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/train_imagenet.py \
  /work/NBB/serihiro/dataset/imagenet/256x256_all/train.ssv \
  /work/NBB/serihiro/dataset/imagenet/256x256_all/val.ssv \
  --mean /work/NBB/serihiro/dataset/imagenet/256x256_all/mean.npy \
  --root /work/NBB/serihiro/dataset/imagenet/256x256_all/train \
  --arch resnet50 \
  --loaderjob 2 \
  --batchsize 32 \
  --val_batchsize 32 \
  --epoch 1 \
  --out /work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/result_prefetch_intel2 \
  --communicator pure_nccl

