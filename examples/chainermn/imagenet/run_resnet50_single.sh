#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -l elapstim_req=12:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH
#PBS -v NQSV_MPI_VER=3.1.4/gcc-cuda10.1
#PBS -T openmpi
#PBS -b 1

NQSV_MPI_VER=3.1.4/intel-cuda10.1
module load cuda/10.1
module load cudnn/7.5.0/10.1
module load openmpi/$NQSV_MPI_VER

# mpirun -np 4 -x LD_PRELOAD=/work/NBB/serihiro/local/lib/libucp.so \
#  /work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/train.sh

mpirun ${NQSII_MPIOPTS} -np 4 -x LD_PRELOAD=/work/NBB/serihiro/local/lib/libucp.so \
  /work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/train.sh

