#!/bin/bash

. ./common.sh

[ -n "$1" ] && CKPT_BEST=$1
[ -z "$CKPT_BEST" ] && exit

echo "Using checkpoint $CKPT_DIR/$CKPT_BEST!"

CMD="python freeze.py $COMMON_OPTS --checkpoint $CKPT_DIR/${CKPT_BEST//_bnfused/} --output_file $GRAPH_FILE"
echo "Running: $CMD"
$CMD
