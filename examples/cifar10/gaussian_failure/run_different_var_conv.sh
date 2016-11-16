#!/bin/bash

MEAN=${MEAN:-5000000}
stds=(500000 1000000 1500000 2000000 2500000 3000000)
DEVICE_IDS=(0 1 2 3)
prob=5
index=0
for std in ${stds[*]}; do
    echo "MEAN: $MEAN; STD: $std; prob of +-1: $prob. Run on GPU ${DEVICE_IDS[$index]}"
    python run_gaussian_exp.py -y --prob $prob --tag _var_conv_exp_prob_$prob $MEAN $std ${DEVICE_IDS[$index]} >/dev/null &
    index=$(((index + 1)%4))
done
