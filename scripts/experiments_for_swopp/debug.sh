#!/bin/bash

source /work/NBB/serihiro/venv/default/bin/activate

python /work/NBB/serihiro/src/chainer/scripts/experiments_for_swopp/scripts/evaluate_prefetch_multiprocess_iterator.py \
--train /work/NBB/serihiro/dataset/imagenet/256x256_all/train.ssv --root /work/NBB/serihiro/dataset/imagenet/256x256_all/train --local_storage_base /scr/local_storage_base --batch_size 32 --n_prefetch 1000 --n_prefetch_from_backend 2 --n_generate_batch 2 --n_remove_example 1 --count 10
