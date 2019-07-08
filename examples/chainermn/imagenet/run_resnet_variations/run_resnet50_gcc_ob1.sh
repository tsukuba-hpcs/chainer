#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 2
#PBS -l elapstim_req=12:00:00
#PBS -v NQSV_MPI_VER=3.1.4/gcc-cuda10.1
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
module load openmpi/$NQSV_MPI_VER

echo ${NQSII_MPIOPTS}

mpirun ${NQSII_MPIOPTS} -mca pml ob1 \
  -np 8 -npernode 4 \
  /work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/train.sh

