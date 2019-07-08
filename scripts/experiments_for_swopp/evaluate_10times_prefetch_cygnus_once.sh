#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -l elapstim_req=6:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

ROOT="/work/NBB/serihiro/src/chainer/scripts/experiments_for_swopp"
config_base="${ROOT}/configs/prefetch_multiprocess_iterator"
log="${ROOT}/resulsts/evaluate_10times_prefetch_multiprocess_iterator_once"
log_stderr="${ROOT}/logs/evaluate_10times_prefetch_multiprocess_iterator_once/`date "+%Y%m%d_%H%M%S"`"

rm -rf $log
mkdir -p $log
mkdir -p ${log_stderr}

mkdir -p /scr/local_storage_base
touch /work/NBB/serihiro/dummy
cp /work/NBB/serihiro/dummy /scr/local_storage_base
rm -rf /scr/local_storage_base/*

config="${config_base}/config_32_12_2_1000.json"

python "${ROOT}/scripts/evaluate_prefetch_multiprocess_iterator.py" \
`config2args "${config}"` \
1>> "${log}/12" \
2>> "${log_stderr}/12"

