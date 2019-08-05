#!/bin/bash

WORK_DIR=work/DS_CNN
mkdir -p $WORK_DIR

TRAIN_DIR=$WORK_DIR/training
LOG_DIR=$WORK_DIR/logs
HYPEROPT_DIR=$WORK_DIR/hyperopt

CKPT_DIR=$TRAIN_DIR/best
CKPT_BEST=$(grep -w model_checkpoint_path $CKPT_DIR/checkpoint | cut -d'"' -f2) 2>/dev/null

DATA_DIR=../speech_commands_v0.02

WANTED_WORDS="yes,no,up,down,left,right,on,off,stop,go"

MODEL=ds_cnn
GRAPH_FILE=$WORK_DIR/${MODEL}.pb

MODEL_SIZE_INFO="5 64 10 4 2 2 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1"
DCT_CNT=10
WIN_SIZE=40
WIN_STRIDE=20
CLIP_DURATION=1000

TESTING_PERCENTAGE=10
VALIDATION_PERCENTAGE=10
TRAINING_PERCENTAGE=80
UNKNOWN_PERCENTAGE=50

ACT_MAX="0 0 0 0 0 0 0 0 0 0 0 0"

HYPER_OPTS=" \
  --model_size_info $MODEL_SIZE_INFO \
  --dct_coefficient_count $DCT_CNT \
  --window_size_ms $WIN_SIZE \
  --window_stride_ms $WIN_STRIDE "

COMMON_OPTS=" \
  --data_url= \
  --data_dir=$DATA_DIR \
  --wanted_words $WANTED_WORDS \
  --clip_duration_ms $CLIP_DURATION \
  --model_architecture $MODEL \
  --testing_percentage $TESTING_PERCENTAGE \
  --validation_percentage $VALIDATION_PERCENTAGE \
  --training_percentage $TRAINING_PERCENTAGE \
  --unknown_percentage $UNKNOWN_PERCENTAGE \
  $HYPER_OPTS "

TRAIN_OPTS=" \
  $COMMON_OPTS \
  --summaries_dir $LOG_DIR --train_dir $TRAIN_DIR "

# 100% for validation and 50% for training
HYPEROPT_OPTS=" \
  $COMMON_OPTS \
  --validation_percentage 100 \
  --training_percentage 50 \
  --hyperopt_dir $HYPEROPT_DIR "

QUANT_OPTS=" \
  $COMMON_OPTS \
  --act_max $ACT_MAX "
