from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.platform import flags
import scipy.io as si
import time
import gc


import logging
import numpy as np
# import pdb
import os
from cleverhans.attacks_SAE import SmoothCarliniWagnerL2
from cleverhans.utils import AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from basic_cnn_models import make_basic_cnn

FLAGS = flags.FLAGS

def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=6,
                      batch_size=128, nb_classes=10, source_samples=100,
                      learning_rate=0.001, attack_iterations=100,
                      targeted=False, alpha='0.800000', namuda='10'):
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

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_test, Y_test = data_mnist(test_start=test_start,
                                   test_end=test_end)

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
    # train_params = {
    #     'nb_epochs': nb_epochs,
    #     'batch_size': batch_size,
    #     'learning_rate': learning_rate,
    #     'train_dir': "./models",
    #     'filename': "basic_cnn.ckpt"
    # }

    # rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    tf_model_load(sess, '../models/basic_cnn.ckpt')

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
    print("This could take some time ...")

    # Instantiate a CW attack object

    start = time.clock()
    cw = Clip_version_old(model, back='tf', sess=sess)
    adv = cw.generate_np(adv_inputs, A, At
                         **cw_params)
    # pdb.set_trace()
    for A_i in range(100):
        save_path = "../dataset/A"
        file_name = "mnist_"+str(lamubda)+"_"+str(alpha)+"_"+str(A_i)+".mat"
        save_name = os.path.join(save_path,file_name)
        A = si.loadmat(save_name)
        adv_A = A['adv_A']
        adv_At = A['adv_At']

        start = time.time()
        cw_params = {'binary_search_steps': 1,
                     'confidence': FLAGS.confidence,
                     yname: adv_ys,
                     'max_iterations': attack_iterations,
                     'learning_rate': 0.1,
                     'batch_size': source_samples if
                     targeted else source_samples,
                     'initial_const': FLAGS.initial_const,
                     'A': adv_A,
                     'At': adv_At}

        # get x and y
        start_p = (A_i)*100
        end_p = (A_i+1)*100
        adv_inputs = np.array(X_test[start_p:end_p], dtype=np.float32)
        adv_inputs = adv_inputs.reshape(
            (source_samples, img_rows, img_cols, 1))
        adv_y = Y_test[start_p:end_p]
        pre_adv = sess.run(preds, feed_dict={x: adv, y: adv_y})

        elapsed = (time.time() - start)
        file_name = "smooth/minst_BasicCnn_clipl2_" + \
            str((A_i+1)*100)+str(FLAGS.confidence)+"_"+str(FLAGS.initial_const)+"_adv_x.mat"
        save_path = os.path.join(data_save_dir1, file_name)
        si.savemat(save_path, {'adv_x': adv})
        file_name = "smooth/minst_BasicCnn_clipl2_" + \
            str((A_i+1)*100)+str(FLAGS.confidence)+"_"+str(FLAGS.initial_const)+"_adv_pre.mat"
        save_path = os.path.join(data_save_dir1, file_name)
        si.savemat(save_path, {'adv_pre': pre_adv})
        eval_params = {'batch_size': source_samples}
        adv_accuracy = 1 - model_eval(sess, x, y, preds, adv, adv_y, args=eval_params)

        print('--------------------------------------')

        print("Time used:", elapsed)
        # Compute the number of adversarial examples that were successfully found
        print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
        report.clean_train_adv_eval = 1. - adv_accuracy

        # Compute the average distortion introduced by the algorithm
        percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                           axis=(1, 2, 3))**.5)
        print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
        gc.collect()
        gc.garbage
    end = time.clock()
    print(end-start)
    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
   # if viz_enabled:
   #     import matplotlib.pyplot as plt
   #     _ = grid_visual(grid_viz_data)

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
                      alpha=FLAGS.alpha,
                      namuda=FLAGS.namuda)


if __name__ == '__main__':
    import argparse
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 100, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_boolean('attack_iterations', 100,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', False,
                         'Run the tutorial in targeted mode?')

    parser = argparse.ArgumentParser()
    parser.add_argument("namuda", help="namuda for matrix")
    parser.add_argument("alpha", help="alpha for matrix")
    parser.add_argument("confidence", help="confidence for cw attack")
    parser.add_argument("initial_const", help="initial_const for cw attack")
    args = parser.parse_args()
    flags.DEFINE_string('namuda', args.namuda, 'namuda for smooth matrix')
    flags.DEFINE_float('alpha', args.alpha, 'alpha for smooth matrix')
    flags.DEFINE_float('confidence', args.confidence, 'confidence for cw attack')
    flags.DEFINE_float('initial_const', args.initial_const, 'initial_const for cw attack')

    tf.app.run()
