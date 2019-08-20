#!/bin/bash

. ./common.sh

CMD="python generate_streaming_test_wav.py $COMMON_OPTS \
  --unknown_percentage 50 \
  --output_audio_file streaming_test.wav \
  --output_labels_file streaming_test_labels.txt"
echo "Running: $CMD"
$CMD
