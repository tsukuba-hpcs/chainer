#!/bin/bash

config_base="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/configs/1000"
log="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/result_prefetch_10times_reversed_16.log"

CONFIGS=(\
"${config_base}/config_32_1000_16_8_1_1000.json" \
"${config_base}/config_32_1000_16_4_1_1000.json" \
"${config_base}/config_32_1000_16_2_1_1000.json"\
)

source /home/serizawa/python36_test/prefetch_iterator/chainer/venv/bin/activate

for config in ${CONFIGS[@]}
do
    /home/serizawa/.cargo/bin/config2args "${config}" >> ${log}
    for _i in `seq 10`
    do
        rm -rf /work/imagenet/local_storage_base/*
        /ppxsvc/bin/ppx-drop-caches
        python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
        `/home/serizawa/.cargo/bin/config2args "${config}"` \
        >> ${log}
    done
    echo '' >> ${log}
done

