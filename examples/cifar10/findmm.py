import os
import sys
sys.path.insert(0, 
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))),
                             'python'))
import caffe

net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)
for name, blobvec in net.params.iteritems():
    for index, blob in enumerate(blobvec):
        print "{:<10} {:<4}: min: {}, max: {}\n".format(name, index, blob.data.min(), blob.data.max())
