from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import glob
import numpy as np
import scipy.io as si
from PIL import Image

def load_imagenet_list(imageDir):
    image = []
    imagelist = glob.glob(os.path.join(imageDir,'*.png'))
    for item in imagelist:
        image.append(os.path.basename(item))
    return image

def load_imagenet_image(file_name):
    img = np.array(Image.open(file_name).convert('RGB')).astype(np.float) / 255.0
    img = img * 2.0 - 1.0
    return img

