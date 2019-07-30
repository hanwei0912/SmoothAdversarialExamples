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
from knn import construct_mnist_graph
#from cleverhans.utils_mnist import data_mnist
from load_data import *

def imagenet():
    path_name = "../../dataset/images" # path of your data
    lamubda = 300
    alpha   = 0.997
    imgnames = load_imagenet_list(path_name)
    for imgname in imgnames:
        pdb.set_trace()
        imgpath =os.path.join(path_name,imgname)
        img=load_imagenet_image(imgpath)
        A = construct_imagenet_graph(img,lamubda,alpha)
        save_path = "../../dataset/A" # path of your saving archive
        file_name = imgname+"_"+str(lamubda)+"_"+str(alpha)+".mat"
        save_name = os.path.join(save_path,file_name)
        si.savemat(save_name,{'A':A})
    return

def mnist():
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)
    lamubda = 10
    alpha   = 0.95
    eig_num = 300
    shape = X_test.shape
    for i in range(10):
        A = np.ones((100,shape[1]*shape[2],eig_num),dtype=float)
        At = np.ones((100,eig_num,shape[1]*shape[2]),dtype=float)
        for j in range(100): # save in 100 batch in case the memory exhaust
            ind = i*100 +j
            img = X_test[ind]
            adv_A, adv_At = construct_mnist_graph(img,lamubda,alpha,eig_num)
            A[j] = adv_A
            At[j]= adv_At
        save_path = "../dataset/A"
        file_name = "mnist_"+str(lamubda)+"_"+str(alpha)+"_"+str(i)+".mat"
        save_name = os.path.join(save_path,file_name)
        si.savemat(save_name,{'adv_A':A,'adv_At':At})
    return


if __name__ == '__main__':
    imagenet()
