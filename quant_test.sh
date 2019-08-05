#!/bin/bash

. ./common.sh

[ -n "$1" ] && CKPT_BEST=$1
[ -z "$CKPT_BEST" ] && exit

echo "Using checkpoint $CKPT_DIR/$CKPT_BEST!"

function fix_header()
{
    [ "$MODEL" = "ds_cnn" ] || return

    sed -i "s/DS-CNN_fc1/FINAL_FC/" $1
    sed -i "s/_depthwise//" $1
    sed -i "s/weights_0/WT/" $1
    sed -i "s/biases_0/BIAS/" $1

    for i in $(seq 1 10); do
        sed -i "s/DS-CNN_conv_ds_${i}_pw_conv/CONV$(($i + 1))_PW/" $1
        sed -i "s/DS-CNN_conv_ds_${i}_dw_conv/CONV$(($i + 1))_DS/" $1
    done

    sed -i "s/DS-CNN_conv_1/CONV1/" $1
}

CMD="python quant_test.py $QUANT_OPTS --checkpoint $CKPT_DIR/$CKPT_BEST"
echo "Running: $CMD"
$CMD && fix_header weights.h
