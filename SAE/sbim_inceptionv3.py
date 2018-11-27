from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pdb

from cleverhans.attacks_hw import SmoothBasicIterativeMethodCG
import numpy as np
from PIL import Image
from cleverhans.utils_tf import tf_model_load
# from PIL import Image
from cleverhans.model import Model
import scipy.io as si
import time

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path',
    '/nfs/pyrex/raid6/hzhang/2017-nips/fgsm/inception_v3.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/test/Timages',
    'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir',
    '/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/test/Timages/sbim',
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
    '/nfs/pyrex/raid6/hzhang/2017-nips/dev_dataset.csv',
    'Path to metadata file.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, metadata_file_path,  batch_shape):
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
    labels = np.zeros(batch_shape[0], dtype=np.int32)
    row_idx_true_label = header_row.index('TrueLabel')
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')
                             ).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        row = rows[idx]
        labels[idx] = int(row[row_idx_true_label])
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images, labels
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images, labels


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.
    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


class InceptionModel(Model):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        self.logits = end_points['Logits']
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        self.probs = output.op.inputs[0]
        return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    # eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)
    # path_name = "/nfs/pyrex/raid6/hzhang/SmoothPerturbation/time/time_log.txt"
    # f = open(path_name, "a")
    # f.write("ImageNet,InceptionModel,clip_verion_debug,")
    # f.write("100,1,16,20,")

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

    bim = SmoothBasicIterativeMethodCG(model,back='tf',sess=sess)
    eps=5/255.0
# 2 epsi 8 iter 5  0.548
# 4 epsi 10 iter 5 0.553
# 4 epsi 10 iter 6
# 2 epsi 8 iter 3
    bim_params = {'eps': 10,
               'ord':2,
               'eps_iter':3,
               'clip_min': -1.,
               'clip_max': 1.}
    adv_x = bim.generate(x_input, adv_A, **bim_params)
    # Run computation

    for filenames, images, labels in load_images(FLAGS.input_dir, FLAGS.metadata_file_path, batch_shape):
        y_labels = np.zeros((FLAGS.batch_size, num_classes))
        for i_y in range(FLAGS.batch_size):
            y_labels[i_y][labels[i_y]] = 1
        # load matrix A

        def load_A(filenames, batch_shape, img_size, namuda):
            #data_dir = "/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/imagenet/A/SpA_"
            data_dir = "/nfs/pyrex/raid6/hzhang/2017-nips/test_A/SpA_"
            n_f = len(filenames)
            A = np.zeros((batch_shape, 4, 299, 299, 3))
            for i in range(n_f):
                name_p = data_dir+filenames[i]+"_"+namuda+"_0.950000.mat"
                data = si.loadmat(name_p)
                A[i] = data['A']
            return A
        A = load_A(filenames, FLAGS.batch_size,
                   FLAGS.image_height*FLAGS.image_width, '10.000000')
        A = np.array(A, dtype=np.float32)


        #preds_adv = model.get_probs(adv_x) 
        x_adv = sess.run(adv_x,feed_dict={x_input:images,adv_A:A})
        print('done 16')
        save_images(x_adv, filenames, FLAGS.output_dir)
    end = time.clock()
    print(end - start)
    # f.write("%s," % str(end-start))
    # f.write("%s," % str(cw.iteration*16/1000))
    # f.write("%s," % str(cw.terminate_situation))

    # f.close()


if __name__ == '__main__':
    tf.app.run()
