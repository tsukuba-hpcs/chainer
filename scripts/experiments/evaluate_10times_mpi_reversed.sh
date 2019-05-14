#!/bin/bash

config_base="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_multiprocess_iterator/configs/1000"
log="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/result_mpi_10times_reversed.log"

date >> ${log}

CONFIGS=(\
"${config_base}/config_ssd_32_16_1000.json" \
"${config_base}/config_ssd_32_12_1000.json" \
"${config_base}/config_ssd_32_8_1000.json" \
"${config_base}/config_ssd_32_4_1000.json" \
"${config_base}/config_ssd_32_2_1000.json" \
"${config_base}/config_nfs_32_16_1000.json" \
"${config_base}/config_nfs_32_12_1000.json" \
"${config_base}/config_nfs_32_8_1000.json" \
"${config_base}/config_nfs_32_4_1000.json" \
"${config_base}/config_nfs_32_2_1000.json"\
)

source /home/serizawa/python36_test/prefetch_iterator/chainer/venv/bin/activate

for config in ${CONFIGS[@]}
do
    /home/serizawa/.cargo/bin/config2args "${config}" >> ${log}
    for _i in `seq 10`
    do
        rm -rf /work/imagenet/local_storage_base/*
        /ppxsvc/bin/ppx-drop-caches
        python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
        `/home/serizawa/.cargo/bin/config2args "${config}"` \
        >> ${log}
    done
    echo '' >> ${log}
done

