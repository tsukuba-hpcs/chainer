#!/bin/bash

config_base="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/configs/1000"
log="./result_mpi_10times_reversed.log"

CONFIGS=(\
"${config_base}/config_32_1000_8_16_1_1000.json" \
"${config_base}/config_32_1000_8_12_1_1000.json" \
"${config_base}/config_32_1000_8_8_1_1000.json" \
"${config_base}/config_32_1000_8_4_1_1000.json" \
"${config_base}/config_32_1000_8_2_1_1000.json"\
)

# CONFIGS=("${config_base}/config_32_1000_8_2_1_1000.json")

for config in ${CONFIGS[@]}
do
    echo $config
    config2args "${config}" >> ${log}
    for _i in `seq 10`
    do
        rm -rf /work/imagenet/local_storage_base/*
        /ppxsvc/bin/ppx-drop-caches
        python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
        `config2args "${config}"` \
        >> ${log}
    done
    echo '' >> ${log}
done

