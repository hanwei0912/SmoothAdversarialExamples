from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pdb

from attacks_SAE import SmoothCarliniWagnerSparse
import numpy as np
from PIL import Image
from cleverhans.utils_tf import tf_model_load
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import scipy.io as si
import time

import logging
from basic_cnn_models import *
from basic_cnn_models import _top_1_accuracy
from load_data import *
from knn import *
from absl import flags, app


slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'checkpoint_path',
    '/nfs/pyrex/raid6/hzhang/2017-nips/models/inception_v3.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '/nfs/pyrex/raid6/hzhang/2017-nips/test/panda',
    #'input_dir', '/nfs/pyrex/raid6/hzhang/2017-nips/images',
    'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir',
    '/nfs/pyrex/raid6/hzhang/2017-nips/clipl2/from_zeros',
    'Output directory with images.')

tf.flags.DEFINE_string(
    'metadata_file_path',
    '/nfs/pyrex/raid6/hzhang/2017-nips/test.csv',
    #'/nfs/pyrex/raid6/hzhang/2017-nips/dev_dataset.csv',
    'Path to metadata file.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

def main(_):
    # ImageNet-specific dimentions
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    lamubda = 10
    alpha = 0.95
    checkpoint_path = FLAGS.checkpoint_path
    input_dir = FLAGS.input_dir
    metadata_file_path = FLAGS.metadata_file_path

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape = batch_shape)
    y_label = tf.placeholder(tf.int32, shape=(batch_shape[0],))
    A       = tf.placeholder(tf.float32)
    model = InceptionModel(num_classes)
    preds = model(x_input)
    logits = model.get_logits(x_input)
    acc = _top_1_accuracy(logits, y_label)
    tf_model_load(sess, checkpoint_path)

    attack = SmoothCarliniWagnerL2Sparse(model, sess=sess)

    for images, _, labels, filenames in load_images(input_dir, input_dir, metadata_file_path, batch_shape):
        pred=sess.run(preds,feed_dict={x_input:images, y_label:labels})
        ind=np.argmax(pred)
        #Aa = construct_imagenet_graph((images+1.0)*0.5,lamubda,alpha)
        y_labels = np.zeros((FLAGS.batch_size, num_classes))
        for i_y in range(FLAGS.batch_size):
            y_labels[i_y][labels[i_y]] = 1
        def load_A(filenames, batch_shape, img_size, namuda):
            data_dir = "/nfs/pyrex/raid6/hzhang/2017-nips/test_A/SpA_"
            n_f = len(filenames)
            A = np.zeros((batch_shape, 4, 299, 299, 3))
            for i in range(n_f):
                name_p = data_dir+filenames[i]+"_"+namuda+"_0.997000.mat"
                data = si.loadmat(name_p)
                A[i] = data['A']
            return A
        A = load_A(filenames, FLAGS.batch_size,
                   FLAGS.image_height*FLAGS.image_width, '300.000000')
        A = np.array(A, dtype=np.float32)
        adv_params = {'binary_search_steps':9,
                      'max_iterations':100,
                      'learning_rate':1e-2,
                      'initial_const': 10,
                      'batch_size':batch_shape[0],
                      'clip_min': -1.,
                      'clip_max':1.,
                      'alpha': alpha,
                      'flag':True}
        adv_x = attack.generate(x_input, A, **adv_params)
        begin=time.time()
        x_adv = sess.run(adv_x,feed_dict={x_input:images, A:Aa})
        end=time.time()
        print('cost total:', end-begin,'s')
        dist = np.sum((x_adv-images)**2,axis=(1,2,3))**.5
        suc = sess.run(acc, feed_dict={x_input:x_adv,y_label:labels})
        print('distortion:',dist[0],'Is adversarial(0:Yes,1:NO)',suc)

    sess.close()
    return

if __name__ == '__main__':
    app.run(main)
