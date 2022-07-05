#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;

function prompt() {
    echo;
    echo "Syntax: bash train.sh EXP_NAME [GPU_IDS] [TRAIN_CONFIG_NAME] [INITIAL_MODEL] [SOURCE_DATA_DIR]";
    echo "GPU_IDS is optional, by default CPU is used";
    echo "CONFIG_NAME is optional, by default 'default_train_args' is used.";
    echo "NOTE: <CONFIG_NAME>.json file should be in 'configs/pretraining/train_config' directory.";
    echo "INITIAL_MODEL is optional, by default 'codet5-base' is used.";
    echo "SOURCE_DATA_DIR is optional, by default 'processed' is used. Which means the training "
    echo "will assume the processed data in 'pretraining/data/processed' directory."
    exit;
}

while getopts ":h" option; do
    case $option in
        h) # display help
          prompt;
    esac
done

if [[ $# < 1 ]]; then
  prompt;
fi

EXPERIMENT_NAME=$1;
GPUS=${2:-"-1"};
CONFIG_NAME=${3:-"default_train_args"};
INITIAL_MODEL=${4:-"codet5-base"};
SOURCE_DATA_DIR=${5:-"processed"};

NUM_GPUS=`echo $(($(echo "${GPUS}" | grep -o "," | wc -l)+1))`
# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000);
# Allow multiple threads
export OMP_NUM_THREADS=8;


CODE_HOME_DIR=`realpath ../..`;
DATA_DIR="${CODE_HOME_DIR}/data/pretraining/${SOURCE_DATA_DIR}";
OUTPUT_DIR="${CODE_HOME_DIR}/models/pretrained/${EXPERIMENT_NAME}"
mkdir -p $OUTPUT_DIR;
CONFIG_PATH="${CODE_HOME_DIR}/configs/pretraining/train_config/${CONFIG_NAME}.json"

printf "************************************************************************************************************\n"
printf "Experiment Name : ${EXPERIMENT_NAME}\n"
printf "GPUS            : ${GPUS}\n"
printf "Training config : ${CONFIG_PATH}\n"
printf "Input Data From : ${DATA_DIR}\n"
printf "Output Model To : ${OUTPUT_DIR}\n"
printf "************************************************************************************************************\n"


export PYTHONPATH=$PYTHONPATH:$CODE_HOME_DIR;
python_script_path="${CODE_HOME_DIR}/src/pretraining/pretrain.py"
export CUDA_VISIBLE_DEVICES=${GPUS};

if [[ ${NUM_GPUS} -lt 2 ]]; then
    python ${python_script_path} \
        --training_config ${CONFIG_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --initial_model ${INITIAL_MODEL} \
        --workers 25 \
        --data_path ${DATA_DIR} 2>&1 | tee ${OUTPUT_DIR}/results.out;
else
    python -m torch.distributed.launch \
        --nproc_per_node ${NUM_GPUS} \
        --master_port $PORT_ID \
        ${python_script_path} \
        --training_config ${CONFIG_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --initial_model ${INITIAL_MODEL} \
        --workers 25 \
        --data_path ${DATA_DIR} 2>&1 | tee ${OUTPUT_DIR}/results.out;
fi


