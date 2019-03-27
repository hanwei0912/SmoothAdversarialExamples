
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import pickle
import csv
import scipy.io as si
import time 

import logging
import os
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans_tutorials.tutorial_models import make_basic_cnn

FLAGS = flags.FLAGS


def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=False, nb_epochs=6,
                      batch_size=128, nb_classes=10, source_samples=10,
                      learning_rate=0.001, attack_iterations=100,
                      targeted=False, confidence=0,initial_const=10):
    """
    MNIST tutorial for Carlini and Wagner's attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    data_save_dir = "/nfs/pyrex/raid6/hzhang/SmoothPerturbation"
    # Get MNIST test data
    #X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
    #                                              train_end=train_end,
    #                                              test_start=test_start,
    #                                              test_end=test_end)

    # keep the test data in same order
    file_name = "X_test.mat"
    load_path = os.path.join(data_save_dir,file_name)
    data=si.loadmat(load_path)
    X_test = data['X_test']
    file_name = "Y_test.mat"
    load_path = os.path.join(data_save_dir,file_name)
    data=si.loadmat(load_path)
    Y_test = data['Y_test']

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    model = make_basic_cnn()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################
    
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': "./models",
        'filename': "basic_cnn_v1.ckpt"
    }

    rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    tf_model_load(sess,'models/basic_cnn_adv.ckpt')
    eval_params = {'batch_size': batch_size}

    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################

    # Instantiate a CW attack object
    cw = CarliniWagnerL2(model, back='tf', sess=sess)

    adv_inputs = np.array(X_test,dtype=np.float32)
    yname = "y"
    adv_ys  = Y_test

    start = time.time()
    cw_params = {'binary_search_steps': 1,
                 'max_iterations': attack_iterations,
                 'learning_rate': 0.1,
                 'batch_size': 100,
                 'initial_const': initial_const,
                 'confidence': confidence}

    adv = cw.generate_np(adv_inputs,
                         **cw_params)
    #file_name = "mnist_BasicCnn_cw_"+str(confidence)+"_"+str(initial_const)+"_adv_x.mat"
    #save_path = os.path.join(data_save_dir,file_name)
    si.savemat('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/update/mnist_advTrain_cw_adv_x.mat',{'adv_x':adv})
    pre_adv = sess.run(preds,feed_dict={x:adv,y:Y_test})
    # evaluate the data we need
    elapsed = (time.time() - start)

    eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    adv_accuracy = 1 - \
        model_eval(sess, x, y, preds, adv, Y_test, args=eval_params)

    print('-------------------------------------')
    print("Time used:",elapsed)
    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1. - adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                       axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
    #file_name = "mnist_BasicCnn_cw_"+str(confidence)+"_"+str(initial_const)+"_adv_pre.mat"
    #save_path = os.path.join(data_save_dir,file_name) 
    #si.savemat(save_path,{'adv_pre':pre_adv})
    # Close TF session
    sess.close()


    return report


def main(argv=None):
    mnist_tutorial_cw(viz_enabled=FLAGS.viz_enabled,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      nb_classes=FLAGS.nb_classes,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate,
                      attack_iterations=FLAGS.attack_iterations,
                      targeted=FLAGS.targeted,
                      confidence=FLAGS.confidence,
                      initial_const=FLAGS.initial_const)


if __name__ == '__main__':
    import argparse
    flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
    flags.DEFINE_boolean('attack_iterations', 100,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', False,
                         'Run the tutorial in targeted mode?')
    parser = argparse.ArgumentParser()
    parser.add_argument("confidence",help="confidence for cw attack")
    parser.add_argument("initial_const",help="initial_const for cw attack")
    args = parser.parse_args()
    flags.DEFINE_float('confidence', args.confidence,'confidence for cw attack')
    flags.DEFINE_float('initial_const', args.initial_const,'initial_const for cw attack')

    tf.app.run()
