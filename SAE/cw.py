from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pdb
import time

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
import numpy as np
from PIL import Image
from cleverhans.model import Model
import scipy.io as si
from basic_cnn_models import InceptionModel, _top_1_accuracy
from load_data import load_images, save_images

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

FLAGS = tf.flags.FLAGS

def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    y_label = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
    y_hot   = tf.one_hot(y_label, num_classes)

    model = InceptionModel(num_classes)
    preds = model(x_input)
    logits = model.get_logits(x_input)
    acc = _top_1_accuracy(logits, y_label)
    tf_model_load(sess, FLAGS.checkpoint_path)

    attack = CarliniWagnerL2(model, back='tf', sess=sess)
    params = {'binary_search_steps':9,
              'y': None,
              'max_iterations':100,
              'learning_rate':0.01,
              'batch_size':1,
              'initial_const':1,
              'clip_min':-1.,
              'clip_max':1.}
    l2 = np.zeros((1000,1))
    acc_ori = np.zeros((1000,1))
    acc_val = np.zeros((1000,20))
    adv_image = np.zeros((1000,299,299,3))
    b_i = 0
    name = []
    begin=time.time()
    for images, _, labels, filenames in load_images(FLAGS.input_dir, FLAGS.input_dir, FLAGS.metadata_file_path, batch_shape):
        bb_i = b_i + FLAGS.batch_size
        name.append(filenames[0])
        y_labels = np.zeros((FLAGS.batch_size,num_classes))
        for i_y in range(FLAGS.batch_size):
            y_labels[i_y][labels[i_y]]=1
        #x_adv = sess.run(adv_x,feed_dict={x_input:images,y_label:labels})
        x_adv = attack.generate_np(images, **params)
        adv_image[b_i]=x_adv
        l2[b_i]=np.mean(np.sum((images- x_adv)**2,axis=(1,2,3))**.5)
        b_i = bb_i
        save_images(x_adv, filenames, FLAGS.output_dir)
    print(time.time()-begin)
    path_save="/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/new/inception/cw/0.01.mat"
    si.savemat(path_save,{'adv':adv_image,'l2':l2,'name':name})


if __name__ == '__main__':
    tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')
    tf.flags.DEFINE_string(
        'checkpoint_path', '../models/inception_v3.ckpt', 'Path to checkpoint for inception network.')
    tf.flags.DEFINE_string(
        'input_dir', '/nfs/pyrex/raid6/hzhang/2017-nips/images', 'Input directory with images.')
    path_save = '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/new/inception/cw/0.01'
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
        'batch_size', 1, 'How many images process at one time.')
    tf.flags.DEFINE_string(
        'metadata_file_path',
        '/nfs/pyrex/raid6/hzhang/2017-nips/dev_dataset.csv',
        'Path to metadata file.')
    tf.app.run()
