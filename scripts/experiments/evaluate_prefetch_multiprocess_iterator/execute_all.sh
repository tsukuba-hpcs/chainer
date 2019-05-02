#!/bin/bash

log_base=./logs
mkdir -p $log_base
log_file_name=${log_base}/`date "+%Y%m%d%H%M%S"`.log
touch $log_file_name
configs=`ls -a ./configs/config*`

for config in $configs
do
    echo $config
    args=`config2args ${config}`
    echo ${args}
    /ppxsvc/bin/ppx-drop-caches
    elapsed_time=`python evaluate_prefetch_multiprocess_iterator.py ${args}`
    echo "${config},${elapsed_time}" >> ${log_file_name}
done
cp -pr ${log_file_name} ${log_base}/latest.log

