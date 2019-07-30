#!/bin/bash

source /Users/kazuhiroserizawa/Documents/python-work/chainer/venv/bin/activate

rm -rf /tmp/local_storage_base
mkdir -p /tmp/local_storage_base

python /Users/kazuhiroserizawa/Documents/python-work/chainer/scripts/experiments_for_swopp/scripts/evaluate_prefetch_multiprocess_iterator.py \
--train /Users/kazuhiroserizawa/Documents/datasets/imagenet-subset/train/train.ssv \
--root /Users/kazuhiroserizawa/Documents/datasets/imagenet-subset/train/1000_train \
--local_storage_base /tmp/local_storage_base \
--batch_size 32 \
--n_prefetch 1000 \
--n_prefetch_from_backend 2 \
--n_generate_batch 2 \
--count 1

rm -rf /tmp/local_storage_base
