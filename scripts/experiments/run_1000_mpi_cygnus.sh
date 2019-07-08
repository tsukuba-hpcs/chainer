#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -l elapstim_req=12:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

log="/work/NBB/serihiro/src/chainer/scripts/experiments/result_mpi_1000_cygnus.log"
echo '' >> $log
date >> ${log}
for config in `ls -a /work/NBB/serihiro/src/chainer/scripts/experiments/evaluate_multiprocess_iterator/configs_cygnus/1000/*.json`
do
    config2args $config >> ${log}
    python /work/NBB/serihiro/src/chainer/scripts/experiments/evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
    `config2args "${config}"` \
    >> ${log}
    echo '' >> ${log}
done

