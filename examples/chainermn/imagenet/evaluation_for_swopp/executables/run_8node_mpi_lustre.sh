#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 8
#PBS -M nserihiro+cygnus@gmail.com
#PBS -m e
#PBS -l elapstim_req=6:00:00
#PBS -v NQSV_MPI_VER=3.1.4/intel-cuda10.1
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

ROOT="/work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp"

module load cuda/10.1
module load cudnn/7.5.0/10.1
module load openmpi/$NQSV_MPI_VER

touch /work/NBB/serihiro/dummy
mkdir -p /scr/local_storage_base
cp /work/NBB/serihiro/dummy /scr/local_storage_base

current_datetime=`date +%Y%m%d_%H%M%S`
time mpirun ${NQSII_MPIOPTS} \
    -x UCX_MAX_RNDV_LANES=4 \
    -np 32 -npernode 4 \
    ${ROOT}/train_multiprocess_iterator_lustre.sh 32 ${current_datetime}

