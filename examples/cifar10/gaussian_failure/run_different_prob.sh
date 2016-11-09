#!/bin/bash

MEAN=5000000
STD=3100000
DEVICE_IDS=(1 2 3)
index=0
for prob in `seq 1 4 24`; do
    echo "MEAN: $MEAN; STD: $STD; prob of +-1: $prob. Run on GPU ${DEVICE_IDS[$index]}"
    python run_gaussian_exp.py -y --prob $prob --tag _exp_prob_$prob $MEAN $STD ${DEVICE_IDS[$index]} >/dev/null &
    index=$(((index + 1)%3))
done
