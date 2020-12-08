from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings
import time
from utils_SAE import *

def main(argv=None):
    shape = [1,299,299,3]
    A = tf.placeholder(tf.float32, (1,4, 299, 299,3))
    p = tf.placeholder(tf.float32, (1,299,299,3))

    smo_mod = CG(A,p,shape)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    g=tf.gradients(smo_mod,p)

    Aa = np.random.random((1,4,299,299,3))
    pp = np.random.random((1,299,299,3))

    begin = time.time()
    sess.run(g,feed_dict={A:Aa,p:pp})
    end   = time.time()



    print('time cost',end-begin,'s')




if __name__ == '__main__':
    tf.app.run()
