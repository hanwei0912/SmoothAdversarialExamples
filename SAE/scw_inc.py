from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pdb
import time

from cleverhans.attacks_SAE import SmoothCarliniWagnerL2CG
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
import numpy as np
from PIL import Image
from cleverhans.model import Model
import scipy.io as si
from basic_cnn_models import InceptionModel
from load_data import load_images, save_images

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

FLAGS = tf.flags.FLAGS

def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)
    preds = model(x_input)
    tf_model_load(sess, FLAGS.checkpoint_path)

    cw = SmoothCarliniWagnerL2CG(model, sess=sess)
    # Run computation

    adv_image= np.zeros((1000,299,299,3))
    l2=np.zeros((1000))
    b_i=0
    name = []
    for images, _, labels, filenames in load_images(FLAGS.input_dir, FLAGS.input_dir, FLAGS.metadata_file_path, batch_shape):
        y_labels = np.zeros((FLAGS.batch_size,num_classes))
        for i_y in range(FLAGS.batch_size):
            y_labels[i_y][labels[i_y]]=1
        # load matrix A
        def load_A(filenames, batch_shape, img_size, namuda):
            data_dir = "/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/A/SpA_"
            n_f = len(filenames)
            A = np.zeros((batch_shape, 4, 299, 299, 3))
            for i in range(n_f):
                name_p = data_dir+filenames[i]+"_"+namuda+"_0.997000.mat"
                data = si.loadmat(name_p)
                A[i] = data['A']
            return A
        A = load_A(filenames, FLAGS.batch_size, FLAGS.image_height*FLAGS.image_width, '300.000000')
        A = np.array(A, dtype=np.float32)

        name.append(filenames[0])
        start = time.time()
        cw_params = {'binary_search_steps': FLAGS.binary_search_steps,
                     'y': None,
                     'max_iterations': FLAGS.max_iteration,
                     'learning_rate': FLAGS.learning_rate,
                     'batch_size': FLAGS.batch_size,
                     'initial_const': FLAGS.initial_const,
                     'clip_min': -1.,
                     'clip_max': 1,
                     'A': A}
        x_adv = cw.generate_np(images,
                               **cw_params)
        elapsed = (time.time() - start)
        print("Time used:", elapsed)
        save_images(x_adv, filenames, FLAGS.output_dir)
        adv_image[b_i:b_i+20]=x_adv
        l2[b_i:b_i+20]=np.sum((images- x_adv)**2,axis=(1,2,3))**.5
        b_i=b_i+20
        path_save="/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/new/inception/scw/adv/real/"+str(FLAGS.learning_rate)+str(b_i/20)+".mat"
        si.savemat(path_save,{'adv':adv_image,'l2':l2,'name':name})


if __name__ == '__main__':
    #import argparse
    #parser = argparse.ArgumentParser()
    ## parser.add_argument("confidence", help="confidence for cw attack")
    ## parser.add_argument("initial_const", help="initial_const for cw attack")
    ## parser.add_argument("max_iteration", help="max_iteration for cw attack")
    ## parser.add_argument("binary_search_steps", help="binary_search_steps for cw attack")
    #parser.add_argument("learning_rate", help="learning_rate for cw attack")
    #args = parser.parse_args()
    tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')
    # tf.flags.DEFINE_float('confidence', args.confidence, 'confidence for cw attack')
    # tf.flags.DEFINE_float('initial_const', args.initial_const, 'initial_const for cw attack')
    # tf.flags.DEFINE_integer('max_iteration', args.max_iteration, 'max_iteration for cw attack')
    # tf.flags.DEFINE_integer('binary_search_steps', args.binary_search_steps,
    #                      'binary_search_steps for cw attack')
    tf.flags.DEFINE_float('confidence', 1, 'confidence for cw attack')
    tf.flags.DEFINE_float('initial_const', 1, 'initial_const for cw attack')
    tf.flags.DEFINE_integer('max_iteration', 100, 'max_iteration for cw attack')
    tf.flags.DEFINE_integer('binary_search_steps', 9,
                         'binary_search_steps for cw attack')
    tf.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate for cw attack')
    #tf.flags.DEFINE_float('learning_rate', args.learning_rate, 'learning_rate for cw attack')
    tf.flags.DEFINE_string(
        'checkpoint_path', '/nfs/pyrex/raid6/hzhang/2017-nips/models/ens_adv_inception/adv/adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')
    tf.flags.DEFINE_string(
        'input_dir', '/nfs/pyrex/raid6/hzhang/2017-nips/images', 'Input directory with images.')
    path_save ='/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/new/inception/scw/adv/'+str(FLAGS.learning_rate)
    folder = os.path.exists(path_save)
    if not folder:
        os.makedirs(path_save)
    tf.flags.DEFINE_string(
        'output_dir', path_save, 'Output directory with images.')
    tf.flags.DEFINE_float(
        'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
    tf.flags.DEFINE_integer(
        'image_width', 299, 'Width of each input images.')
    tf.flags.DEFINE_integer(
        'image_height', 299, 'Height of each input images.')
    tf.flags.DEFINE_integer(
        'batch_size', 20, 'How many images process at one time.')
    tf.flags.DEFINE_string(
        'metadata_file_path',
        '../dataset/dev_dataset.csv',
        'Path to metadata file.')
    tf.app.run()
