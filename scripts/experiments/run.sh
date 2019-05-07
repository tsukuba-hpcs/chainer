#!/bin/bash

echo 'multiprocess_iterator'
/ppxsvc/bin/ppx-drop-caches
python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
`config2args evaluate_multiprocess_iterator/configs/config_ssd_32_16_10.json` \
> result.log

echo 'prefetch_multiprocess_iterator'
/ppxsvc/bin/ppx-drop-caches
python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
`config2args evaluate_prefetch_multiprocess_iterator/configs/config_32_10_8_16_1.json` \
>> result.log

