#!/bin/bash
# try threshold method

MEAN=${MEAN:-5000000}
STD=${STD:-500000}
DEVICE_IDS=(0 1 2 3)
prob=5
index=0
thresholds=(0.00001 0.000001 0.0000001 0.00000001)

for t in ${thresholds[*]}; do
    echo "MEAN: $MEAN; STD: $STD; prob of +-1: $prob; threshold strategy: ${t}. Run on GPU ${DEVICE_IDS[$index]}"
    python run_gaussian_exp.py -y -t $t --prob $prob --tag _conv_exp_prob_$prob $MEAN $STD ${DEVICE_IDS[$index]} >/dev/null &
    index=$(((index + 1)%4))
done
