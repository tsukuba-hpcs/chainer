#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -l elapstim_req=6:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

ROOT="/work/NBB/serihiro/src/chainer/scripts/experiments_for_swopp"
config_base="${ROOT}/configs/prefetch_multiprocess_iterator"
log="${ROOT}/resulsts/evaluate_10times_prefetch_multiprocess_iterator"

mkdir -p $log

CONFIGS=(\
"${config_base}/config_32_2_1_1000.json" \
"${config_base}/config_32_2_2_1000.json" \
"${config_base}/config_32_2_4_1000.json" \
"${config_base}/config_32_2_8_1000.json" \
"${config_base}/config_32_2_12_1000.json" \
"${config_base}/config_32_2_16_1000.json"\
)

N_PROCESSES=(1 2 4 8 12 16)

rm -rf $log
mkdir -p $log

index=0
for config in ${CONFIGS[@]}
do
    for i in `seq 10`
    do
        python "${ROOT}/scripts/evaluate_prefetch_multiprocess_iterator.py" \
        `config2args "${config}"` >> \
        "${log}/${N_PROCESSES[$index]}"
    done
    index=$(expr $index + 1)
done

