#!/bin/bash

module load cudnn/7.5.0/10.1

source /work/NBB/serihiro/venv/default/bin/activate


/usr/sbin/dropcaches 3

python /work/NBB/serihiro/src/chainer/scripts/experiments_for_swopp/scripts/evaluate_multiprocess_iterator.py \
--train /work/NBB/serihiro/dataset/imagenet/256x256_all/train.ssv \
--root /work/NBB/serihiro/dataset/imagenet/256x256_all/train \
--batch_size 32 \
--n_prefetch 1000 \
--n_processes 2 \
--count 10

