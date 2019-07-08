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
log="${ROOT}/result_mpi_10times_cygnus_ssd.log"

echo 'start cp files'
time cp /work/NBB/serihiro/dataset/imagenet/256x256_all.tar.gz /scr
cd /scr
echo 'start tar xzf files'
time tar xzf /scr/256x256_all.tar.gz
cd /scr/256x256_all
echo 'start create labels'
python2 labeling.py train val

#CONFIGS=(\
#"${config_base}/config_ssd_32_2_1000.json" \
#"${config_base}/config_ssd_32_4_1000.json" \
#"${config_base}/config_ssd_32_8_1000.json" \
#"${config_base}/config_ssd_32_12_1000.json" \
#"${config_base}/config_ssd_32_16_1000.json"\
#)

CONFIGS=("${config_base}/config_ssd_32_16_1000.json")

for config in ${CONFIGS[@]}
do
    echo 'start python'
    python "${ROOT}/evaluate_multiprocess_iterator/evaluate_multiprocess_iterator.py" \
    `config2args "${config}"`
done

