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
log="/work/NBB/serihiro/src/chainer/scripts/experiments/result_prefetch_10times.log"

CONFIGS=(\
"${config_base}/config_32_1000_8_2_1_1000.json" \
"${config_base}/config_32_1000_8_4_1_1000.json" \
"${config_base}/config_32_1000_8_8_1_1000.json" \
"${config_base}/config_32_1000_8_12_1_1000.json" \
"${config_base}/config_32_1000_8_16_1_1000.json"\
)

for config in ${CONFIGS[@]}
do
    config2args "${config}" >> ${log}
    for i in `seq 10`
    do
	echo $i
        rm -rf /scr/local_storage_base/*
	python /work/NBB/serihiro/src/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
        `config2args "${config}"` \
        >> ${log}
    done
    echo '' >> ${log}
done

