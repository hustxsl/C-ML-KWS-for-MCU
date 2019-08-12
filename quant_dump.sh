#!/bin/bash

. ./common.sh

[ -n "$1" ] && CKPT_BEST=$1
[ -z "$CKPT_BEST" ] && exit

echo "Using checkpoint $CKPT_DIR/$CKPT_BEST!"

CMD="python quant_dump.py $QUANT_OPTS --checkpoint $CKPT_DIR/$CKPT_BEST"
echo "Running: $CMD"
$CMD && [ "$MODEL" = "ds_cnn" ] && mv ds_cnn* Deployment/Source/NN_C/DS_CNN/
