#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -m e
#PBS -M nserihiro+cygnus@gmail.com
#PBS -l elapstim_req=02:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

ROOT="/work/NBB/serihiro/src/chainer/scripts/experiments_for_swopp"
config_base="${ROOT}/configs/prefetch_multiprocess_iterator"
log="${ROOT}/resulsts/evaluate_10times_prefetch_multiprocess_iterator_once"
log_stderr="${ROOT}/logs/evaluate_10times_prefetch_multiprocess_iterator_once/`date "+%Y%m%d_%H%M%S"`"

mkdir -p $log
mkdir -p ${log_stderr}

mkdir -p /scr/local_storage_base
touch /work/NBB/serihiro/dummy
cp /work/NBB/serihiro/dummy /scr/local_storage_base


config="${config_base}/config_32_2_2_1000.json"
rm -rf /scr/local_storage_base/*
/usr/sbin/dropcaches 3

python "${ROOT}/scripts/evaluate_prefetch_multiprocess_iterator.py" \
`config2args "${config}"` \
1> "${log}/2" \
2>> "${log_stderr}/2"


config="${config_base}/config_32_4_2_1000.json"
rm -rf /scr/local_storage_base/*
/usr/sbin/dropcaches 3

python "${ROOT}/scripts/evaluate_prefetch_multiprocess_iterator.py" \
`config2args "${config}"` \
1> "${log}/4" \
2>> "${log_stderr}/4"


config="${config_base}/config_32_8_2_1000.json"
rm -rf /scr/local_storage_base/*
/usr/sbin/dropcaches 3

python "${ROOT}/scripts/evaluate_prefetch_multiprocess_iterator.py" \
`config2args "${config}"` \
1> "${log}/8" \
2>> "${log_stderr}/8"


config="${config_base}/config_32_12_2_1000.json"
rm -rf /scr/local_storage_base/*
/usr/sbin/dropcaches 3

python "${ROOT}/scripts/evaluate_prefetch_multiprocess_iterator.py" \
`config2args "${config}"` \
1> "${log}/12" \
2>> "${log_stderr}/12"


config="${config_base}/config_32_16_2_1000.json"
rm -rf /scr/local_storage_base/*
/usr/sbin/dropcaches 3

python "${ROOT}/scripts/evaluate_prefetch_multiprocess_iterator.py" \
`config2args "${config}"` \
1> "${log}/16" \
2>> "${log_stderr}/16"

