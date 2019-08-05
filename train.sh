#!/bin/bash

. ./common.sh

EXTRA_OPTS=
if [ "$1" = "continue" ]; then
    shift
    [ -n "$CKPT_BEST" ] && \
    EXTRA_OPTS=" --start_checkpoint $CKPT_DIR/${CKPT_BEST//_bnfused/}"
fi

if [ "$1" = "quick" ]; then
    shift
    EXTRA_OPTS="$EXTRA_OPTS --eval_step_interval 100"
fi

CMD="python train.py $TRAIN_OPTS $EXTRA_OPTS"
echo "Running: $CMD"
$CMD
