#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -v NQSV_MPI_VER=3.1.4/intel-cuda10.1
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

exec {my_fd}< "$0"
echo $my_fd
flock -n ${my_fd}
if [  $? -eq 0 ]; then
    echo 'I got the lock!! YEAH!!!'
    sleep 5
else
    echo 'I cannot get the lock...wtf'
fi

