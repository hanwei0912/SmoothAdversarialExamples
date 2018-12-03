from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pdb
import glob
import numpy as np
import scipy.io as si
from PIL import Image

path_name = "/nfs/pyrex/raid6/hzhang/2017-nips/images/"

def load_imagenet(imageDir):
    image = []
    imgname = []
    imagelist = glob.golb(os.path.join(imageDir,'*.png'))
    for item in imagelist:
        image.append(os.path.basename(item))
    for item in image:
        (temp1,)=os.path.splitext(item)
        imgname.append(temp1)
    return image, imgname

def similarity(x_img,lamuda):
    shape = x_img.shape()
    im_mod = np.pad(x_img, ((1,1),(1,1)),'constant',constant_values=0)
    im_r = im_mod[1:shape[0]+1,0:shape[0]]
    im_l = im_mod[]
