#!/bin/bash

log="result_run2.log"
config="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/configs/1000/config_32_1000_8_16_1_1000.json"

date >> ${log}
rm -rf /work/imagenet/local_storage_base/*
/ppxsvc/bin/ppx-drop-caches
python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
`config2args "${config}"` \
>> ${log}

