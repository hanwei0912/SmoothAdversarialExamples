from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import pdb
import os
import unittest
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception
from cleverhans.utils_tf import tf_model_load
import scipy.io as si

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.model import Model

DEFAULT_INCEPTION_PATH = (
    '/nfs/pyrex/raid6/hzhang/2017-nips/models/inception_v3.ckpt')
    #'/nfs/pyrex/raid6/hzhang/2017-nips/models/ens_adv_inception/adv/adv_inception_v3.ckpt')

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path',
    DEFAULT_INCEPTION_PATH, 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_image_dir',
    #'/udd/hzhang/SmoothAdversarialExamples/SAE/new',
    '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/new/inception/cw/b0.01',
    'Path to image directory.')
    #'/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/bilateral/0.5-0.2-our',
    #'/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/inceptionV3/FGSM/1',

tf.flags.DEFINE_string(
    'origin_image_dir',
    #'/udd/hzhang/SmoothAdversarialExamples/dataset/zibra',
    '/nfs/pyrex/raid6/hzhang/2017-nips/images',
    'Path to image directory.')
    #'/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/bilateral/0.5-0.2',

tf.flags.DEFINE_string(
    'metadata_file_path',
    #'/nfs/pyrex/raid6/hzhang/2017-nips/test.csv',
    '/nfs/pyrex/raid6/hzhang/2017-nips/dev_dataset.csv',
    'Path to metadata file.')

FLAGS = tf.flags.FLAGS

def load_images(input_dir, ori_input_dir, metadata_file_path,  batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array,
      i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    with open(metadata_file_path) as input_file:
        reader = csv.reader(input_file)
        header_row = next(reader)
        rows = list(reader)

    images = np.zeros(batch_shape)
    X_test = np.zeros(batch_shape)
    labels = np.zeros(batch_shape[0], dtype=np.int32)
    rows = np.array(rows)
    row_idx_true_label = header_row.index('TrueLabel')
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')
                             ).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        ori_filepath = os.path.join(ori_input_dir,os.path.basename(filepath))
        with tf.gfile.Open(ori_filepath) as f:
            ori_image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
        X_test[idx,:,:,:]=ori_image
        (name,_) = os.path.splitext(os.path.basename(filepath))
        (ind_l,_) = np.where(rows==name)
        row = rows[ind_l][0]
        labels[idx] = int(row[row_idx_true_label])
        idx += 1
        if idx == batch_size:
            yield images, X_test, labels, filenames
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield images, X_test, labels, filenames

class InceptionModel(Model):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            # Inception preprocessing uses [-1, 1]-scaled input.
            x_input = x_input * 2.0 - 1.0
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        self.logits = end_points['Logits']
        # Strip off the extra reshape op at the output
        self.probs = end_points['Predictions'].op.inputs[0]
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)


def _top_1_accuracy(logits, labels):
    return tf.reduce_mean(
        tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))



def main(_):
    """Check model is accurate on unperturbed images."""
    input_dir = FLAGS.input_image_dir
    origin_dir = FLAGS.origin_image_dir
    metadata_file_path = FLAGS.metadata_file_path
    num_images = 1
    batch_shape = (num_images, 299, 299, 3)
    num_classes = 1001
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)

    tf.logging.set_verbosity(tf.logging.INFO)
        # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    y_label = tf.placeholder(tf.int32, shape=(num_images,))
    model = InceptionModel(num_classes)
    logits = model.get_logits(x_input)
    pred   = model.get_probs(x_input)
    acc = _top_1_accuracy(logits, y_label)
    #tf_model_load(sess, FLAGS.checkpoint_path)
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        success=np.zeros((1000,1))
        change=np.zeros((1000,1))
        ori_acc=np.zeros((1000,1))
        l2_norm = np.zeros((1000,1))
        l2_norm_o = np.zeros((1000,1))
        name = []
        b_i=0
        for images, X_test, labels,filename in load_images(input_dir,origin_dir, metadata_file_path, batch_shape):
            #y_labels = np.zeros((num_classes, num_images))
            #for i_y in range(num_images):
            #    y_labels[labels[i_y]][i_y] = 1
            pred_t = sess.run(pred,feed_dict={x_input:X_test,y_label:labels})
            ll=np.array([np.argmax(pred_t)],dtype=np.int32)
            acc_val = sess.run(acc, feed_dict={
                x_input: images, y_label: labels})
            cha_val = sess.run(acc, feed_dict={
                x_input: images, y_label: ll})
            success[b_i]=acc_val
            change[b_i]=cha_val
            name.append(filename[0])
            ori_acc_val = sess.run(acc, feed_dict={
                x_input: X_test, y_label: labels})
            ori_acc[b_i]=ori_acc_val
            #print('acc:%s',acc_val)
            grey = np.ones((1,299,299,3))
            L2_norm = np.mean(np.sum((images-X_test)**2,axis=(1,2,3))**.5)
            L2_norm_o = np.mean(np.sum((grey-X_test)**2,axis=(1,2,3))**.5)
            Li_norm = np.mean(np.max(np.abs(images-X_test),axis=(1,2,3)))
            l2_norm[b_i]=L2_norm
            l2_norm_o[b_i]=L2_norm_o
            #pdb.set_trace()
            b_i=b_i+1
            #print('L2: %s', L2_norm)
            #print('Li: %s', Li_norm)
        si.savemat('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/new/cw_inc_0.01_p_l2_b.mat',{'p':success,'l2':l2_norm,'ori_a':ori_acc,'c':change,'name':name,'l2_worst':l2_norm_o})
        #si.savemat('/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/new/cw_inc_0.01_p_l2.mat',{'p':success,'l2':l2_norm,'ori_a':ori_acc,'c':change,'name':name,'l2_worst':l2_norm_o})




if __name__ == '__main__':
    tf.app.run()
