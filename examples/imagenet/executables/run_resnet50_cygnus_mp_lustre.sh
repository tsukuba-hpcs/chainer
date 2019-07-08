#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -M nserihiro+cygnus@gmail.com
#PBS -m e
#PBS -l elapstim_req=20:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

mkdir -p /scr/local_storage_base
touch /work/NBB/serihiro/dummy
cp /work/NBB/serihiro/dummy /scr/local_storage_base

echo `date +%Y%m%d_%H%M%S`' start dropcaches 3'
/usr/sbin/dropcaches 3
echo `date +%Y%m%d_%H%M%S`' finish dropcaches 3'

config='/work/NBB/serihiro/src/chainer/examples/imagenet/resnet50_config_cygnus/multiprocess_iterator/nfs/32_1000_2.json'
echo `date +%Y%m%d_%H%M%S`' start training'
python /work/NBB/serihiro/src/chainer/examples/imagenet/train_imagenet_extended.py `config2args ${config}`
echo `date +%Y%m%d_%H%M%S`' finish training'

