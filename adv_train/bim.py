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
from cleverhans.attacks_hw import BasicIterativeMethod
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
                   nb_filters=64, num_threads=None,epsi=0.3,epsi_iter=0.08):
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
    bim_params = {'eps': epsi,
                   'eps_iter':epsi_iter,
                   #'ord':2,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        model = make_basic_cnn(nb_filters=nb_filters)
        preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)
#        model_train(sess, x, y, preds, X_train, Y_train,save=True, evaluate=evaluate,
#                    args=train_params, rng=rng)
        tf_model_load(sess,'models/basic_cnn_adv.ckpt')
        #tf_model_load(sess,'models/basic_cnn_fgsm_0.3.ckpt')

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        bim = BasicIterativeMethod(model, sess=sess)
        adv_x = bim.generate(x, **bim_params)
        preds_adv = model.get_probs(adv_x)
        
        # evaluate the data we need
        adv_image =bim.generate_np(X_test[0:10000], **bim_params)
        pre_adv = sess.run(preds_adv,feed_dict={x:X_test[0:10000],y:Y_test[0:10000]})
        name = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/update/mnist_BasicCnn_bim_'+str(epsi)+"_"+str(epsi_iter)+"_adv_x.mat"
        si.savemat(name,{'adv_x':adv_image})
        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_par)
        print('Test accuracy on legetite examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc


    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   epsi=FLAGS.epsi,
                   epsi_iter=FLAGS.epsi_iter)


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
    parser.add_argument("epsi",help="epsi for i-fgsm attack")
    parser.add_argument("epsi_iter",help="epsi_iter for i-fgsm attack")
    args = parser.parse_args()
    flags.DEFINE_float('epsi', args.epsi,'espi for i-fgsm attack')
    flags.DEFINE_float('epsi_iter', args.epsi_iter,'espi_iter for i-fgsm attack')

    tf.app.run()
