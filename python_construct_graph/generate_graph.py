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
from numpy import linalg as LA
from knn import construct_imagenet_graph
from load_data import *

def main():
    path_name = "../dataset/images" # path of your data
    lamubda = 300
    alpha   = 0.997
    imgnames = load_imagenet_list(path_name)
    for imgname in imgnames:
        imgpath =os.path.join(path_name,imgname)
        img=load_imagenet_image(imgpath)
        A = construct_imagenet_graph(img,lamubda,alpha)
        save_path = "../dataset/A" # path of your saving archive
        file_name = imgname+"_"+str(lamubda)+"_"+str(alpha)+".mat"
        save_name = os.path.join(save_path,file_name)
        si.savemat(save_name,{'A':A})
    return

if __name__ == '__main__':
    main()
