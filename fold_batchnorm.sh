#!/bin/bash

. ./common.sh

[ -n "$1" ] && CKPT_BEST=$1
[ -z "$CKPT_BEST" ] && exit

CMD="python fold_batchnorm.py $COMMON_OPTS \
  --checkpoint $CKPT_DIR/${CKPT_BEST//_bnfused/}"
echo "Running: $CMD"
$CMD
