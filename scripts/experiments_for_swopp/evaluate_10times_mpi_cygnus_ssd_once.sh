#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -l elapstim_req=0:15:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

ROOT="/work/NBB/serihiro/src/chainer/scripts/experiments_for_swopp"
config_base="${ROOT}/configs/multiprocess_iterator"
log="${ROOT}/resulsts/evaluate_10times_multiprocess_iterator_ssd_once"

mkdir -p $log

echo 'start copying imagenet'

ls -latr /scr

cp -pr /work/NBB/serihiro/dataset/imagenet/256x256_all.tar.gz /scr
cd /scr
tar xzf 256x256_all.tar.gz
cd /scr/256x256_all
python2 labeling.py train val

ls -latr /scr
ls -latr /scr/256x256_all

echo 'finish copying imagenet'

mkdir -p $log

config=${config_base}/config_ssd_32_12_1000.json

python "${ROOT}/scripts/evaluate_multiprocess_iterator.py" \
`config2args "${config}"` >> \
"${log}/12"

