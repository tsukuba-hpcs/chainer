#!/bin/bash

log="result_mpi_1000.log"
echo '' >> $log
date >> ${log}
echo 'multiprocess_iterator' >> ${log}
for config in `ls -a /home/serizawa/python36_test/prefetch_iterator/chainer/scripts/experiments/evaluate_multiprocess_iterator/configs/1000/*.json`
do
    rm -rf /work/imagenet/local_storage_base/*
    /ppxsvc/bin/ppx-drop-caches
    python evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py \
    `config2args "${config}"` \
    >> ${log}
    echo '' >> ${log}
done

