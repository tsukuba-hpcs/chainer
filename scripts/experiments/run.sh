#!/bin/bash


rm -rf /work/imagenet/local_storage_base/*
echo 'prefetch_multiprocess_iterator' > result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
`config2args evaluate_prefetch_multiprocess_iterator/configs/config_32_10_16_8_1.json` \
>> result.log

rm -rf /work/imagenet/local_storage_base/*
echo 'prefetch_multiprocess_iterator' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
`config2args evaluate_prefetch_multiprocess_iterator/configs/config_32_10_8_16_1.json` \
>> result.log

rm -rf /work/imagenet/local_storage_base/*
echo 'prefetch_multiprocess_iterator' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
`config2args evaluate_prefetch_multiprocess_iterator/configs/config_32_50_16_8_1.json` \
>> result.log

rm -rf /work/imagenet/local_storage_base/*
echo 'prefetch_multiprocess_iterator' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
`config2args evaluate_prefetch_multiprocess_iterator/configs/config_32_50_8_16_1.json` \
>> result.log

echo 'multiprocess_iterator ssd' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args evaluate_multiprocess_iterator/configs/config_ssd_32_8_10.json` \
>>result.log

echo 'multiprocess_iterator ssd' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args evaluate_multiprocess_iterator/configs/config_ssd_32_16_10.json` \
>> result.log

echo 'multiprocess_iterator ssd' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args evaluate_multiprocess_iterator/configs/config_ssd_32_8_50.json` \
>> result.log

echo 'multiprocess_iterator ssd' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args evaluate_multiprocess_iterator/configs/config_ssd_32_16_50.json` \
>> result.log

echo 'multiprocess_iterator nfs' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args evaluate_multiprocess_iterator/configs/config_nfs_32_8_10.json` \
>>result.log

echo 'multiprocess_iterator nfs' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args evaluate_multiprocess_iterator/configs/config_nfs_32_16_10.json` \
>> result.log

echo 'multiprocess_iterator nfs' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args evaluate_multiprocess_iterator/configs/config_nfs_32_8_50.json` \
>> result.log

echo 'multiprocess_iterator nfs' >> result.log
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args evaluate_multiprocess_iterator/configs/config_nfs_32_16_50.json` \
>> result.log
