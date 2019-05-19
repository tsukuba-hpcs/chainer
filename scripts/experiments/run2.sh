#!/bin/bash

log="result_run2.log"
config1="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/configs/1000/config_32_1000_8_16_1_1000.json"

date >> ${log}

echo 'prefetch_multiprocess_iterator' >> ${log}

rm -rf /work/imagenet/local_storage_base/*
/ppxsvc/bin/ppx-drop-caches
python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
`config2args "${config1}"` \
>> ${log}

exit

rm -rf /work/imagenet/local_storage_base/*
/ppxsvc/bin/ppx-drop-caches
python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
`config2args "${config2}"` \
>> ${log}

echo 'multiprocess_iterator (NFS)' >> ${log}

config1="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_multiprocess_iterator/configs/config_nfs_32_16_50.json"
config2="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_multiprocess_iterator/configs/config_nfs_32_8_50.json"

rm -rf /work/imagenet/local_storage_base/*
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args "${config1}"` \
>> ${log}

rm -rf /work/imagenet/local_storage_base/*
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args "${config2}"` \
>> ${log}

echo 'multiprocess_iterator (SSD)' >> ${log}

config1="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_multiprocess_iterator/configs/config_ssd_32_16_50.json"
config2="/home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_multiprocess_iterator/configs/config_ssd_32_8_50.json"

rm -rf /work/imagenet/local_storage_base/*
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args "${config1}"` \
>> ${log}

rm -rf /work/imagenet/local_storage_base/*
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args "${config2}"` \
>> ${log}

