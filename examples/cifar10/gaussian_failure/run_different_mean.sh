#!/bin/bash

means=(0 1000000 3000000)
STD=1500000
DEVICE_IDS=(1 2 3)
#ths=(0.01 0.001 0.0001 0.00001 0.000001 0.0000001)
index=0
for mean in ${means[*]}; do
    echo "MEAN: $mean; STD: $STD; prob of +-1: $prob; Run on GPU ${DEVICE_IDS[$index]}"
    python run_gaussian_exp.py --prob 5 --tag _different_mean_conv_exp_prob_5 $mean $STD ${DEVICE_IDS[$index]}  >/dev/null &
    index=$(((index + 1)%3))
done
