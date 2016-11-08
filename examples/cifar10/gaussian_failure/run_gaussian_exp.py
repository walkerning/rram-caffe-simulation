#!/usr/bin/python
#-*- coding: utf-8 -*-

import argparse
import os
import sys
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("mean", type=float)
parser.add_argument("std", type=float)
parser.add_argument("device_id", type=int)
parser.add_argument("-t", "--threshold", default=-1, type=float)
args = parser.parse_args()

# parsing command line
mean = args.mean
std = args.std
device_id = args.device_id
strategy_suffix="_threshold_{}".format(args.threshold) if args.threshold > 0 else ""

# handle some import/binary paths
here = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(here)))
caffe_bin = os.path.join(root_dir, "build/tools/caffe")
sys.path.insert(0, os.path.join(root_dir, "python"))

# Start working
from caffe.proto import caffe_pb2
from google.protobuf import text_format

SOLVER_DIR = os.path.join(here, "solvers/")
SOLVER_NAME = "solver_{mean}_{std}{strategy}.prototxt"
SNAPSHOT_NAME = "snapshot_{mean}_{std}{strategy}"
TEMPLATE = "solvers/cifar10_vgg11_template.prototxt"

message = caffe_pb2.SolverParameter()
with open(TEMPLATE, "r") as f:
    content = f.read()

text_format.Merge(content, message)
message.failure_pattern.mean = mean
message.failure_pattern.std = std
message.device_id = device_id

snapshot_prefix = SNAPSHOT_NAME.format(mean=mean, std=std, strategy=strategy_suffix)

if os.path.exists(snapshot_prefix):
    while 1:
        yes = raw_input("{} already exists, remove? (y/n): ".format(snapshot_prefix))
        if yes.lower() in {"y", "yes"}:
            assert subprocess.check_call("rm -r {}".format(snapshot_prefix), shell=True) == 0
            break
        elif yes.lower() in {"n", "no"}:
            sys.exit()

assert subprocess.check_call("mkdir -p {}".format(snapshot_prefix), shell=True)  == 0

message.snapshot_prefix = snapshot_prefix + "/"

if args.threshold > 0:
    message.failure_strategy.extend([caffe_pb2.FailureStrategyParameter(type="threshold", threshold=args.threshold)])

solver_fname = os.path.join(SOLVER_DIR, SOLVER_NAME.format(mean=mean, std=std, strategy=strategy_suffix))

# write to new solver file
text_format.PrintMessage(message, open(solver_fname, "w"))
print "New solver prototxt write to {}.".format(solver_fname)

subprocess.check_call("{} train --solver {} 2>&1 | tee {}".format(caffe_bin, 
                                                                  solver_fname,
                                                                  os.path.join(snapshot_prefix, "log")), shell=True)
