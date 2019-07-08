#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -l elapstim_req=12:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

mkdir -p /scr/local_storage_base
touch /work/NBB/serihiro/dummy
cp /work/NBB/serihiro/dummy /scr/local_storage_base

config='/work/NBB/serihiro/src/chainer/examples/imagenet/resnet50_config_cygnus/multiprocess_iterator/nfs/32_1000_16.json'
python /work/NBB/serihiro/src/chainer/examples/imagenet/train_imagenet_extended.py \
    /work/NBB/serihiro/dataset/imagenet/256x256_all/train_1000.ssv \
    /work/NBB/serihiro/dataset/imagenet/256x256_all/val.ssv \
    --local_storage_base /scr/local_storage_base \
    --root /work/NBB/serihiro/dataset/imagenet/256x256_all \
    --mean /work/NBB/serihiro/dataset/imagenet/256x256_all/mean.npy \
    --gpu 0 \
    --arch resnet50 \
    --epoch 2 \
    --local_storage_base /scr/local_storage_base \
    --iterator multiprocess \
    --loaderjob 8 \
    --prefetchjob 16 \
    --n_prefetch 1000 \
    --batchsize 32 \
    --val_batchsize 32 \
    --out /work/NBB/serihiro/src/chainer/examples/imagenet/resnet50_results/prefetch_multiprocess_iterator/32_1000_16_1000
exit 0

