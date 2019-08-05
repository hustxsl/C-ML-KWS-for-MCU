#!/bin/sh

. ./common.sh

EXTRA_OPTS=
if [ "$1" = "quick" ]; then
    shift
    EXTRA_OPTS="$EXTRA_OPTS --how_many_training_steps 3,3,3"
fi

CMD="python hyper_optimize.py $HYPEROPT_OPTS $EXTRA_OPTS"
echo "Running: $CMD"
$CMD
