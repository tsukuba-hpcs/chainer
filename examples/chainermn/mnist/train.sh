#!/bin/bash

export LD_LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

export PATH=/system/apps/cuda/10.1/bin:$PATH
export CPATH=/system/apps/cudnn/7.5.0/cuda10.1/include:/system/apps/cuda/10.1/include:$CPATH
export LIBRARY_PATH=/system/apps/cudnn/7.5.0/cuda10.1/lib64:/system/apps/cuda/10.1/lib64:$LIBRARY_PATH

export CUDA_HOME=/system/apps/cuda/10.1
export CUDA_PATH=/system/apps/cuda/10.1

/work/NBB/serihiro/venv/default/bin/python  /work/NBB/serihiro/src/chainer/examples/chainermn/mnist/train_mnist.py \
  --batchsize 32 \
  --epoch 20 \
  --gpu \
  --communicator pure_nccl

