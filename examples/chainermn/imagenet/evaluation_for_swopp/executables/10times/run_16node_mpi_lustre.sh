#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 16
#PBS -M nserihiro+cygnus@gmail.com
#PBS -m e
#PBS -l elapstim_req=1:00:00
#PBS -v NQSV_MPI_VER=3.1.4/intel-cuda10.1
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

ROOT="/work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp"

NQSV_MPI_VER=3.1.4/intel-cuda10.1
module load cuda/10.1
module load cudnn/7.5.0/10.1
module load openmpi/$NQSV_MPI_VER

mkdir -p /scr/local_storage_base/
touch /work/NBB/serihiro/dummy
cp /work/NBB/serihiro/dummy /scr/local_storage_base/dummy

current_datetime=`date +%Y%m%d_%H%M%S`
time mpirun ${NQSII_MPIOPTS} \
    -x UCX_MAX_RNDV_LANES=4 \
    -np 64 -npernode 4 \
    ${ROOT}/train_multiprocess_iterator_lustre_10times.sh 64 ${current_datetime}

