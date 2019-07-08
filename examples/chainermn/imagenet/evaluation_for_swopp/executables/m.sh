#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 1
#PBS -l elapstim_req=01:00:00
#PBS -v NQSV_MPI_VER=3.1.4/intel-cuda10.1
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

NQSV_MPI_VER=3.1.4/intel-cuda10.1
module load openmpi/$NQSV_MPI_VER

mpirun ${NQSII_MPIOPTS} -np 4 \
    /work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp/executables/mpstat_test.sh

