#!/bin/bash

log="result_1000.log"
echo 'prefetch_multiprocess_iterator' > ${log}
for config in `ls -a /home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/configs/1000/*.json`
do
    rm -rf /work/imagenet/local_storage_base/*
    /ppxsvc/bin/ppx-drop-caches
    python evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
    `config2args "${config}"` \
    >> ${log}
done

