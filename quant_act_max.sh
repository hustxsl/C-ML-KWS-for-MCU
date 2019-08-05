#!/bin/bash

. ./common.sh

[ -n "$1" ] && CKPT_BEST=$1
[ -z "$CKPT_BEST" ] && exit

echo "Using checkpoint $CKPT_DIR/$CKPT_BEST!"

CMD="python quant_act_max.py $QUANT_OPTS --checkpoint $CKPT_DIR/$CKPT_BEST"
echo "Running: $CMD"
$CMD
