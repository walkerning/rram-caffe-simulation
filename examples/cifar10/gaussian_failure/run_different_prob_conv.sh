#!/bin/bash

MEAN=${MEAN:-5000000}
STD=${STD:-3100000}
DEVICE_IDS=(0 1 2 3)
probs=(0 5 25)
index=0
for prob in ${probs[*]}; do
    echo "MEAN: $MEAN; STD: $STD; prob of +-1: $prob. Run on GPU ${DEVICE_IDS[$index]}"
    python run_gaussian_exp.py -y --prob $prob --tag _conv_exp_prob_$prob $MEAN $STD ${DEVICE_IDS[$index]} >/dev/null &
    index=$(((index + 1)%4))
done
