#!/bin/sh
# coding: utf-8
# brief: entry script to try various modes of running

if [ -z "$RUNNER"  ]; then
    echo "Need to set \\$RUNNER"
    exit 1
fi

if [ "$RUNNER" = 0  ]; then
    ./predict.py $@
elif [ "$RUNNER" = 1  ]; then
    ./eval.py $@
elif [ "$RUNNER" = 2  ]; then
    ./train.py $@
else
    echo "Invalid option chosen!"
    exit 1
fi


