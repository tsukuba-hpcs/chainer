#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 1
#PBS -l elapstim_req=12:00:00
#PBS -v NQSV_MPI_VER=3.1.4/intel-cuda10.1
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

if [ $# -ne 2 ];then
    echo 'Usage: this_script {np} {train_script}'
    exit 1
fi

np=${1}
train_script=${2}

ROOT="/work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp"

module load cuda/10.1
module load cudnn/7.5.0/10.1
module load openmpi/$NQSV_MPI_VER

mkdir -p /scr/local_storage_base/
touch /work/NBB/serihiro/dummy
cp /work/NBB/serihiro/dummy /scr/local_storage_base/dummy

time mpirun ${NQSII_MPIOPTS} \
    -x UCX_MAX_RNDV_LANES=4 \
    -np ${1} -npernode 4 \
    ${ROOT}/${train_script}
