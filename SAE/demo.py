from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import scipy.io as si
import time


import logging
import pdb
import os
from attacks_SAE import SmoothBasicIterativeMethod
#from attacks_SAE import SmoothCarliniWagnerL2
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from basic_cnn_models import *
from load_data import *
from knn import *

def mnist_attack():
    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1
    nb_classes = 10
    eig_num = 300
    lamubda = 10
    alpha = 0.95

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    A       = tf.placeholder(tf.float32)

    # Define TF model graph
    pdb.set_trace()
    model = BasicCNNModel()
    preds = model(x)
    print("Defined TensorFlow model graph.")
    tf_model_load(sess,'../models/basic_cnn.ckpt')

    pdb.set_trace()
    attack = SmoothBasicIterativeMethod(model,back='tf', sess=sess)
    adv_params = {'eps': 5,
                  'ord':2,
                  'eps_iter': 3,
                  'flag':False,
                  'clip_min': 0.,
                  'clip_max': 1.}
    adv_x = attack.generate(x, A, ** adv_params)
    rng   = np.random.RandomState([2017,8,30])
    for images, y_label in load_images_m():
        Aa = construct_mnist_graph(img,lamubda,alpha,eig_num)
        x_adv = sess.run(adv_x, feed_dict={x:images, y:y_label, A:Aa})

    sess.close()
    return

def imagnet_attack():
    # ImageNet-specific dimentions
    batch_shape = [1, 299, 299, 3]
    num_classes = 1001
    lamubda = 10
    alpha = 0.95
    checkpoint_path = '../models/inception_v3.ckpt'
    input_dir = '../dataset/images'
    metadata_file_path = '../dataset/dev_dataset.csv'

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape = batch_shape)
    A       = tf.placeholder(tf.float32)
    model = InceptionModel(num_classes)
    preds = model(x_input)
    tf_model_load(sess, checkpoint_path)

    attack = SmoothBasicIterativeMethod(model, sess=sess)
    adv_params = {'eps': 5/255,
                  'ord': 2,
                  'eps_iter': 3,
                  'clip_min': -1.,
                  'clip_max':1.,
                  'flag':True}
    adv_x = attack.generate(x_input, A, **adv_params)

    for images, _, labels, filenames in load_images(input_dir, input_dir, metadata_file_path, batch_shape):
        Aa = construct_imagenet_graph((images+1.0)*0.5,lamubda,alpha)
        begin=time.time()
        x_adv = sess.run(adv_x,feed_dict={x_input:images, A:Aa})
        end=time.time()
        print('cost total:', end-begin,'s')

    sess.close()
    return

def main(_):
    imagnet_attack()

if __name__ == '__main__':
    tf.app.run()


