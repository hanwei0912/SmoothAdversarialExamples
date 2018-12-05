from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pdb

from cleverhans.attacks_SAE import SmoothBasicIterativeMethodCG
import numpy as np
from PIL import Image
from cleverhans.utils_tf import tf_model_load
from cleverhans.model import Model
import scipy.io as si
import time
from basic_cnn_models import InceptionModel
from load_data import load_images, save_images

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path',
    '../models/inception_v3.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '../dataset/images',
    'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir',
    '../dataset/smooth',
    'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_string(
    'metadata_file_path',
    '../dataset/dev_dataset.csv',
    'Path to metadata file.')

FLAGS = tf.flags.FLAGS

def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    start = time.clock()
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    adv_A = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,4,299,299,3))

    model = InceptionModel(num_classes)
    preds = model(x_input)
    tf_model_load(sess, FLAGS.checkpoint_path)

    bim = SmoothBasicIterativeMethodCG(model,sess=sess)
    eps=5/255.0
    bim_params = {'eps': 10,
               'ord':2,
               'eps_iter':3,
               'clip_min': -1.,
               'clip_max': 1.}
    adv_x = bim.generate(x_input, adv_A, **bim_params)
    # Run computation

    for images, _, labels, filenames in load_images(FLAGS.input_dir, FLAGS.input_dir, FLAGS.metadata_file_path, batch_shape):
        y_labels = np.zeros((FLAGS.batch_size, num_classes))
        for i_y in range(FLAGS.batch_size):
            y_labels[i_y][labels[i_y]] = 1
        # load matrix A

        def load_A(filenames, batch_shape, img_size, namuda):
            data_dir = "../dataset/A/"
            n_f = len(filenames)
            A = np.zeros((batch_shape, 4, 299, 299, 3))
            for i in range(n_f):
                name_p = data_dir+filenames[i]+"_"+namuda+"_0.997.mat"
                data = si.loadmat(name_p)
                A[i] = data['A']
            return A
        A = load_A(filenames, FLAGS.batch_size,
                   FLAGS.image_height*FLAGS.image_width, '300')
        A = np.array(A, dtype=np.float32)


        #preds_adv = model.get_probs(adv_x) 
        x_adv = sess.run(adv_x,feed_dict={x_input:images,adv_A:A})
        save_images(x_adv, filenames, FLAGS.output_dir)
    end = time.clock()
    print(end - start)



if __name__ == '__main__':
    tf.app.run()
