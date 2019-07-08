#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -l elapstim_req=12:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate


ROOT="/work/NBB/serihiro/src/chainer/scripts/experiments"
config_base="${ROOT}/evaluate_multiprocess_iterator/configs_cygnus/1000"
log="${ROOT}/result_mpi_10times_cygnus.log"

CONFIGS=(\
"${config_base}/config_nfs_32_2_1000.json" \
"${config_base}/config_nfs_32_4_1000.json" \
"${config_base}/config_nfs_32_8_1000.json" \
"${config_base}/config_nfs_32_12_1000.json" \
"${config_base}/config_nfs_32_16_1000.json"\
)

# CONFIGS=("${config_base}/config_32_1000_8_2_1_1000.json")

for config in ${CONFIGS[@]}
do
    config2args "${config}" >> ${log}
    for _i in `seq 10`
    do
        python "${ROOT}/evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py" \
        `config2args "${config}"` \
        >> ${log}
    done
    echo '' >> ${log}
done

