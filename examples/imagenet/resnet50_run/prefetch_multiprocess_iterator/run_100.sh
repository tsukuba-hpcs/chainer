#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 2
#PBS -l elapstim_req=12:00:00
#PBS -v NQSV_MPI_VER=3.1.4/intel-cuda10.1
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

export NQSV_MPI_VER=3.1.4/intel-cuda10.1

module load cuda/10.1
module load cudnn/7.5.0/10.1
module load openmpi/$NQSV_MPI_VER

ROOT="/work/NBB/serihiro/src/chainer/examples/imagenet"
CONFIG_BASE="${ROOT}/resnet50_config
TRAIN_SCRIPT="${ROOT}/train_imagenet_extended_100times.py"

python ${TRAIN_SCRIPT} `config2args "${CONFIG_BASE}/prefetch_multiprocess_iterator/32_1000_8_16.json"

