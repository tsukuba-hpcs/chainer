#!/bin/bash

source /Users/kazuhiroserizawa/Documents/python-work/chainer/venv/bin/activate

sync && sudo purge

python /Users/kazuhiroserizawa/Documents/python-work/chainer/scripts/experiments_for_swopp/scripts/evaluate_multiprocess_iterator.py \
--train /Users/kazuhiroserizawa/Documents/datasets/imagenet-subset/train/train.ssv \
--root /Users/kazuhiroserizawa/Documents/datasets/imagenet-subset/train/1000_train \
--batch_size 32 \
--n_prefetch 1000 \
--n_processes 2 \
--count 100
