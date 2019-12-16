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
from attacks_SAE import SmoothBasicIterativeMethodDense
from attacks_SAE import SmoothBasicIterativeMethodSparse
from attacks_SAE import SmoothCarliniWagnerSparse
from attacks_SAE import SmoothCarliniWagnerDense
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from basic_cnn_models import *
from basic_cnn_models import _top_1_accuracy
from load_data import *
from knn import *
import matplotlib.pyplot as plt

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
    A       = tf.placeholder(tf.float32, shape = (None, img_rows*img_cols, img_rows*img_cols))

    # Define TF model graph
    model = ModelBasicCNN('model1', 10, 64)
    preds = model.get_probs(x)
    print("Defined TensorFlow model graph.")
    tf_model_load(sess,'../models/mnist')

    #attack = SmoothCarliniWagnerDense(model, sess=sess)
    #adv_params = {'binary_search_steps':9,
    #              'max_iterations':100,
    #              'learning_rate':1e-2,
    #              'initial_const': 10,
    #              'batch_size':1,
    #              'clip_min': -1.,
    #              'clip_max':1.}
    attack = SmoothBasicIterativeMethodDense(model, sess=sess)
    adv_params = {'eps': 9,
                  'ord':2,
                  'eps_iter': 0.8,
                  'flag':False,
                  'clip_min': 0.,
                  'clip_max': 1.}
    adv_x = attack.generate(x, A, ** adv_params)
    rng   = np.random.RandomState([2017,8,30])
    x_test, y_test = data_mnist(test_start=0, test_end=10000)
    for i in range(10000):
        Aa = construct_mnist_graph(x_test[i:i+1],lamubda,alpha)
        x_adv = sess.run(adv_x, feed_dict={x:x_test[i:i+1], y:y_test[i:i+1], A:Aa})
        pre_l = sess.run(preds,feed_dict={x:x_adv, y:y_test[i:i+1]})
        label = np.argmax(y_test[i:i+1])
        pre_l = np.argmax(pre_l)
        l2 = np.sum((x_adv-x_test[i:i+1])**2,axis=(1,2,3))**.5
        is_adv = np.equal(label,pre_l)
        print('distortion:',l2,'is adversarial:',~is_adv)
        pdb.set_trace()
        #implot = plt.imshow(x_adv)

    sess.close()
    return

def imagnet_attack():
    # ImageNet-specific dimentions
    batch_shape = [1, 299, 299, 3]
    num_classes = 1001
    lamubda = 10
    alpha = 0.95
    checkpoint_path = '../models/inception_v3.ckpt'
    input_dir = '../dataset/zibra'
    metadata_file_path = '../dataset/dev_dataset.csv'

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

    #attack = SmoothBasicIterativeMethodSparse(model, sess=sess)
    #adv_params = {'eps': 5/255,
    #              'ord': 2,
    #              'eps_iter': 3,
    #              'clip_min': -1.,
    #              'clip_max':1.,
    #              'alpha': alpha,
    #              'flag':True}
    attack = SmoothCarliniWagnerSparse(model, sess=sess)
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

    for images, _, labels, filenames in load_images(input_dir, input_dir, metadata_file_path, batch_shape):
        Aa = construct_imagenet_graph((images+1.0)*0.5,lamubda,alpha)
        begin=time.time()
        x_adv = sess.run(adv_x,feed_dict={x_input:images, A:Aa})
        end=time.time()
        print('cost total:', end-begin,'s')
        dist = np.sum((x_adv-images)**2,axis=(1,2,3))**.5
        suc = sess.run(acc, feed_dict={x_input:x_adv,y_label:labels})
        print('distortion:',dist[0],'Is adversarial(0:Yes,1:NO)',suc)

    sess.close()
    return

def main(_):
    mnist_attack()

if __name__ == '__main__':
    tf.app.run()


