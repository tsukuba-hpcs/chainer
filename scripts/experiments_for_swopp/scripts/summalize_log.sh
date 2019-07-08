#!/bin/bash

source /work/NBB/serihiro/venv/default/bin/activate

ROOT=/work/NBB/serihiro/src/chainer/scripts/experiments_for_swopp
PROCESSES=(1 2 4 8 12 16)

for p in ${PROCESSES[@]}
do
    python summalize_log.py -l ${ROOT}/logss/evaluate_10times_prefetch_multiprocess_iterator/20190604_102807/${p}  > summalize_log_result_${p}.csv
done
