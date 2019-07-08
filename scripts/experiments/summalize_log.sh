#!/bin/bash

source /work/NBB/serihiro/venv/default/bin/activate

for i in `seq 0 4`; do
  for j in `seq 0 9`; do
    python summalize_log.py -l evaluate_10times_with_profile_prefetch_cygnus_log/log_stderr.${i}.${j}.log >> summalize_log_result_${i}.csv
  done
done

