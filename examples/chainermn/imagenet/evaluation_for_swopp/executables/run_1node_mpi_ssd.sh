#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 1
#PBS -l elapstim_req=12:00:00
#PBS -v NQSV_MPI_VER=3.1.4/intel-cuda10.1
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

ROOT="/work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp"

module load cuda/10.1
module load cudnn/7.5.0/10.1
module load openmpi/$NQSV_MPI_VER

echo 'start copying the dataset'
cp /work/NBB/serihiro/dataset/imagenet/256x256_all.tar.gz /scr
cd /scr
tar xzf 256x256_all.tar.gz
cd 256x256_all
python2 labeling.py train val
echo 'finish copying the dataset'

time mpirun ${NQSII_MPIOPTS} \
    -x UCX_MAX_RNDV_LANES=4 \
    -np 4 -npernode 4 \
    ${ROOT}/train_multiprocess_iterator_ssd.sh 4

