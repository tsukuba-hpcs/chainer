#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 4
#PBS -l elapstim_req=24:00:00
#PBS -v NQSV_MPI_VER=3.1.4/intel-cuda10.1
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load openmpi/3.1.4/intel-cuda10.1

mpirun ${NQSII_MPIOPTS} -np 16 -npernode 4 /work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp/executables/d.sh

