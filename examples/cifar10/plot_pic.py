import re
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

assert len(sys.argv) == 2

with open(sys.argv[1], "r") as log_file:
    content = log_file.read()

test_interval_m = re.search("test_interval: (\d+)", content)
assert test_interval_m is not None
test_interval = int(test_interval_m.group(1))

match_pattern = re.compile(
    r"accuracy = (?P<acc>[\d\.]+).*?loss = (?P<loss>[\d\.]+)",
    re.DOTALL)

acc_list = []
loss_list = []
for match in match_pattern.finditer(content):
    acc_list.append(float(match.group("acc")))
    loss_list.append(float(match.group("loss")))

# print acc_list
# print loss_list

print "iter     accuracy    loss"

iter_indexes = list(np.arange(len(acc_list)) * test_interval)
for data in zip(iter_indexes, acc_list, loss_list):
    print "{:<8}    {:<12}    {:<12}".format(*data)

# plot the pic
if os.environ.get("DISPLAY", None) is not None:
    plt.figure().add_subplot(111).plot(iter_indexes, acc_list, "g")
    plt.figure().add_subplot(111).plot(iter_indexes, loss_list)
    plt.show()

