#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -M nserihiro+cygnus@gmail.com
#PBS -m e
#PBS -l elapstim_req=18:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

mkdir -p /scr/local_storage_base
touch /work/NBB/serihiro/dummy
cp /work/NBB/serihiro/dummy /scr/local_storage_base

# config='/work/NBB/serihiro/src/chainer/examples/imagenet/resnet50_config_cygnus/prefetch_multiprocess_iterator/32_1000_8_16.json'
config='/work/NBB/serihiro/src/chainer/examples/imagenet/resnet50_config_cygnus/prefetch_multiprocess_iterator/32_1000_2_2.json'
python /work/NBB/serihiro/src/chainer/examples/imagenet/train_imagenet_extended.py `config2args ${config}`

