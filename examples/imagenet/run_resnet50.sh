#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -l elapstim_req=12:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

python /work/NBB/serihiro/src/chainer/examples/imagenet/train_imagenet.py \
  /work/NBB/serihiro/dataset/imagenet/256x256_all/train.ssv \
  /work/NBB/serihiro/dataset/imagenet/256x256_all/val.ssv \
  --mean /work/NBB/serihiro/dataset/imagenet/256x256_all/mean.npy \
  --root /work/NBB/serihiro/dataset/imagenet/256x256_all/train \
  --arch resnet50 \
  --device 0 \
  --loaderjob 16 \
  --batchsize 32 \
  --val_batchsize 32 \
  --epoch 1

