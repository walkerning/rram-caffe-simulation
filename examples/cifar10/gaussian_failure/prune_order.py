#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("proto")
parser.add_argument("model")
parser.add_argument("prune_ratio", type=float)
parser.add_argument("output_file")
args = parser.parse_args()

prune_ratio = args.prune_ratio
proto = args.proto
model = args.model
output_file = args.output_file
print "proto: {}; model: {}; prune_ratio: {}; output_file: {}".format(proto, model, prune_ratio, output_file)

# handle some import/binary paths
here = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(here)))
caffe_bin = os.path.join(root_dir, "build/tools/caffe")
sys.path.insert(0, os.path.join(root_dir, "python"))

import caffe
print caffe.__file__

net = caffe.Net(proto, model, caffe.TEST)
fc_weights = []
for key, value in net.params.iteritems():
    if key[:2] == "fc":
        weights = value[0].data
        flatten_data = weights.flatten()
        rank = np.argsort(np.abs(flatten_data))
        flatten_data[rank[:int(rank.size * prune_ratio)]] = 0
        np.copyto(weights, flatten_data.reshape(weights.shape))
        fc_weights.append(weights)

net.save(output_file)

with open(output_file, "w") as wf:
    for i in range(1, len(fc_weights)):
        flag_func = np.vectorize(lambda x: 1 if x == 0 else 0)
        zero_nums = flag_func(fc_weights[i-1]).sum(axis=1) + flag_func(fc_weights[i]).sum(axis=0)
        indexes = np.argsort(zero_nums)
        wf.write(" ".join([str(x) for x in indexes]))
        wf.write("\n")
