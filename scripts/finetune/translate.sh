function prompt() {
    echo "Syntax: bash translate.sh <GPU-ID> <FROM_LANG> <TO_LANG>";
    exit;
}

if [[ $# -ne 3 ]]; then
    prompt;
fi

GPU=$1;
FROM=${2};
TO=${3};
CODE_HOME_DIR=`realpath ../..`;
DATA_DIR="${CODE_HOME_DIR}/data/finetuning";

function download() {
    mkdir -p ${DATA_DIR}/translate;
    cdir=`pwd`;
    cd $DATA_DIR/translate;
    ## Add the code code downloading the data;
    cd ${cdir};
}

function finetune() {
    OUTPUT_DIR="${CODE_HOME_DIR}/models/finetuning/translate";
    mkdir -p ${OUTPUT_DIR};
    LOG="${OUTPUT_DIR}/finetuning.log";
    SUMMARY_DIR="${OUTPUT_DIR}/summary";
    mkdir -p ${SUMMARY_DIR};
    CACHE_DIR="${OUTPUT_DIR}/cache";
    mkdir -p ${CACHE_DIR};
    RES_DIR="${OUTPUT_DIR}/results";
    mkdir -p $RES_DIR;

    PRETEAINED_MODEL_BASE="${CODE_HOME_DIR}/models/pretrained/";
    PRETRAINING_EXP_NAME="unidir"
    PRETRAINED_MODEL_NAME="ckpt-5000-toufiq";
    PRETRAINED_MODEL_PATH="${PRETEAINED_MODEL_BASE}/${PRETRAINING_EXP_NAME}/${PRETRAINED_MODEL_NAME}";

    export PYTHONIOENCODING=utf-8;
    export PYTHONPATH=$PYTHONPATH:$CODE_HOME_DIR;
    SCRIPT_PATH="${CODE_HOME_DIR}/src/finetuning/generation.py";

    export CUDA_VISIBLE_DEVICES=${GPU};

    BATCH_SIZE=32;
    GRADIENT_ACCUM_STEP=16;
    NUM_EPOCHS=30;
    PATIENCE=15;
    LEARNING_RATE=5e-5;
    SRC_LEN=320;
    TGT_LEN=128;

    python $SCRIPT_PATH \
            --do_train --do_eval --do_eval_bleu --do_test \
            --task translate --sub_task ${FROM}-${TO} \
            --model_type codet5 \
            --data_num -1  \
            --warmup_steps 500 \
            --learning_rate ${LEARNING_RATE} \
            --num_train_epochs ${NUM_EPOCHS} \
            --patience ${PATIENCE} \
            --tokenizer_name ${PRETRAINED_MODEL_PATH}  \
            --model_name_or_path ${PRETRAINED_MODEL_PATH} \
            --data_dir ${DATA_DIR}  \
            --cache_path ${CACHE_DIR}  \
            --output_dir ${OUTPUT_DIR}  \
            --summary_dir ${SUMMARY_DIR} \
            --save_last_checkpoints \
            --always_save_model \
            --res_dir ${RES_DIR} \
            --res_fn ${RES_DIR}/results.txt \
            --train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUM_STEP} \
            --eval_batch_size ${BATCH_SIZE} \
            --max_source_length ${SRC_LEN} \
            --max_target_length ${TGT_LEN} 2>&1 | tee ${LOG}
}

download;
finetune;
