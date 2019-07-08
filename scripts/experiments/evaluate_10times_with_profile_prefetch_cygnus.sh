#!/bin/bash

#PBS -A NBB
#PBS -q gen_S
#PBS -M nserihiro+cygnus@gmail.com
#PBS -m e
#PBS -l elapstim_req=3:00:00
#PBS -v LD_LIBRARY_PATH=/work/NBB/serihiro/local/lib:/work/NBB/serihiro/local/lib64:$LD_LIBRARY_PATH

module load cuda/10.1
module load cudnn/7.5.0/10.1
source /work/NBB/serihiro/venv/default/bin/activate

mkdir -p /scr/local_storage_base/
touch /work/NBB/serihiro/dummy
cp /work/NBB/serihiro/dummy /scr/local_storage_base/dummy

CURRENT_DATE=`date +%Y%m%d%H%M%S`

config_base="/work/NBB/serihiro/src/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/configs/1000_cygnus"
log="/work/NBB/serihiro/src/chainer/scripts/experiments/evaluate_10times_with_profile_prefetch_cygnus_log/${CURRENT_DATE}_log_stdout"
log_stderr="/work/NBB/serihiro/src/chainer/scripts/experiments/evaluate_10times_with_profile_prefetch_cygnus_log/${CURRENT_DATE}_log_stderr"

# CONFIGS=(\
# "${config_base}/config_32_1000_8_16_1_1000.json"\
# )

CONFIGS=(\
"${config_base}/config_32_8_2_1000.json" \
"${config_base}/config_32_8_4_1000.json" \
"${config_base}/config_32_8_8_1000.json" \
"${config_base}/config_32_8_12_1000.json" \
"${config_base}/config_32_8_16_1000.json"\
)

i=0
for config in ${CONFIGS[@]}
do
    for j in `seq 0 10`
    do
      echo "${j} start"
      config2args "${config}" > "${log_stderr}.${i}.${j}.log"
      echo 'start rm -rf /scr/local_storage_base/*'
      rm -rf /scr/local_storage_base/*
      echo 'finish rm -rf /scr/local_storage_base/*'
      echo 'start python'
      python /work/NBB/serihiro/src/chainer/scripts/experiments/evaluate_prefetch_multiprocess_iterator/evaluate_prefetch_multiprocess_iterator.py \
      `config2args "${config}"` \
	 >> "${log}.${i}.${j}.log" 2>> "${log_stderr}.${i}.${j}.log"
      echo 'finish python'
      echo '' >> "${log}.${i}.${j}.log"
      echo '' >> "${log_stderr}.${i}.${j}.log"
      echo "${j} done"
    done
    echo $((i++))
done

