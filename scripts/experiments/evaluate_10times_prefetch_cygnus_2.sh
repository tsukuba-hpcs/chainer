#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -l elapstim_req=12:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

mkdir -p /scr/local_storage_base/
touch /work/NBB/serihiro/dummy
cp /work/NBB/serihiro/dummy /scr/local_storage_base/dummy

config_base="/work/NBB/serihiro/src/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/configs/1000_cygnus"
log="/work/NBB/serihiro/src/chainer/scripts/experiments/result_prefetch_10times_cygnus_2.log"
log_stderr="/work/NBB/serihiro/src/chainer/scripts/experiments/result_prefetch_10times_cygnus_2_stderr.log"

CONFIGS=(\
"${config_base}/config_32_1000_20_2_1_1000.json" \
"${config_base}/config_32_1000_18_2_1_1000.json" \
"${config_base}/config_32_1000_16_2_1_1000.json" \
"${config_base}/config_32_1000_14_2_1_1000.json" \
"${config_base}/config_32_1000_12_2_1_1000.json" \
"${config_base}/config_32_1000_10_2_1_1000.json" \
"${config_base}/config_32_1000_8_2_1_1000.json" \
"${config_base}/config_32_1000_6_2_1_1000.json" \
"${config_base}/config_32_1000_4_2_1_1000.json" \
"${config_base}/config_32_1000_2_2_1_1000.json"\
)

for config in ${CONFIGS[@]}
do
    config2args "${config}" >> ${log}
    config2args "${config}" >> ${log_stderr}
    for i in `seq 10`
    do
        rm -rf /scr/local_storage_base/*
	python /work/NBB/serihiro/src/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
        `config2args "${config}"` \
        >> ${log} 2>> ${log_stderr}
    done
    echo '' >> ${log}
    echo '' >> ${log_stderr}
done

