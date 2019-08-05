#!/bin/bash

. ./common.sh

CMD="python label_wav.py --wav test.wav --graph $GRAPH_FILE --labels $TRAIN_DIR/ds_cnn_labels.txt --how_many_labels 1"
echo "Running: $CMD"
$CMD
