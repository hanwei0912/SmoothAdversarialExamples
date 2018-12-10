from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pdb
import glob
import numpy as np
import scipy.io as si
from PIL import Image
from cleverhans import dataset
import tempfile


def load_imagenet_list(imageDir):
    image = []
    imagelist = glob.glob(os.path.join(imageDir,'*.png'))
    for item in imagelist:
        image.append(os.path.basename(item))
    return image

def load_imagenet_image(file_name):
    img = np.array(Image.open(file_name).convert('RGB')).astype(np.float) / 255.0
    #img = img * 2.0 - 1.0
    return img

def maybe_download_mnist_file(file_name, datadir=None, force=False):
  url = os.path.join('http://yann.lecun.com/exdb/mnist/', file_name)
  return dataset.maybe_download_file(url, datadir=None, force=False)


def download_and_parse_mnist_file(file_name, datadir=None, force=False):
  return dataset.download_and_parse_mnist_file(file_name, datadir=None,
                                               force=False)


def data_mnist(datadir=tempfile.gettempdir(), train_start=0,
               train_end=60000, test_start=0, test_end=10000):
  mnist = dataset.MNIST(train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        center=False)
  return mnist.get_set('train') + mnist.get_set('test')

