"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
import csv
import pickle
import pdb
import scipy.io as si

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans.attacks_hw import FastGradientMethod
from basic_cnn_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level

import os

FLAGS = flags.FLAGS


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None,epsi=0.3):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    data_save_dir = "/nfs/pyrex/raid6/hzhang/SmoothPerturbation"
    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    # keep the test data in same order
    file_name = "X_test.mat"
    load_path = os.path.join(data_save_dir,file_name)
    data=si.loadmat(load_path)
    X_test = data['X_test']
    file_name = "Y_test.mat"
    load_path = os.path.join(data_save_dir,file_name)
    data=si.loadmat(load_path)
    Y_test = data['Y_test']

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model_path = "models/mnist"

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': "./models",
        'filename': "basic_cnn.ckpt"
    }

    rng = np.random.RandomState([2017, 8, 30])

    model = make_basic_cnn(nb_filters=nb_filters)
    preds = model.get_probs(x)
    #tf_model_load(sess,'models/basic_cnn_fgsm_0.3.ckpt')
    tf_model_load(sess,'models/basic_cnn_adv.ckpt')

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test
        # examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, x, y, preds, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Calculate training error
    if testing:
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, x, y, preds, X_train, Y_train, args=eval_params)
        report.train_clean_train_clean_eval = acc

    fgsm = FastGradientMethod(model, sess=sess)

    fgsm_params = {'eps': epsi,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_images =fgsm.generate_np(X_test, **fgsm_params)
    preds_adv = model.get_probs(adv_x)
    eval_par = {'batch_size': 128}
    # evaluate the data we need
    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_par)
    print('Test accuracy on legitimate examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc

    # Calculate training error
    if testing:
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_train,
                         Y_train, args=eval_par)
        report.train_clean_train_adv_eval = acc
    save_path_name ='/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/update/mnist_BasicCnn_fgsm_'+str(epsi)+'_adv_x.mat'
    si.savemat(save_path_name,{'adv_x':adv_images})
    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   epsi = FLAGS.epsi)


if __name__ == '__main__':
    import argparse
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    parser = argparse.ArgumentParser()
    parser.add_argument("epsi",help="epsi for fgsm attack")
    args = parser.parse_args()
    flags.DEFINE_float('epsi', args.epsi,'espi for fgsm attack')
    tf.app.run()
