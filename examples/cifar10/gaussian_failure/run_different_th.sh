#!/bin/bash

MEAN=5000000
STD=1500000
DEVICE_IDS=(1 2 3)
ths=(0.01 0.001 0.0001 0.00001 0.000001 0.0000001)
index=0
for th in ${ths[*]}; do
    echo "MEAN: $MEAN; STD: $STD; prob of +-1: $prob; th: $th; Run on GPU ${DEVICE_IDS[$index]}"
    python run_gaussian_exp.py --prob 5 --tag _second_th_conv_exp_prob_5_$th  $MEAN $STD ${DEVICE_IDS[$index]} -t $th  >/dev/null &
    index=$(((index + 1)%3))
done
