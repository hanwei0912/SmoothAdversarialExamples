from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import csv
from numpy import genfromtxt
import scipy.io as si
import time


import logging
import pdb
import os
from cleverhans.attacks_hw import Clip_version
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from basic_cnn_models import make_basic_cnn

data_save_dir = "/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/update"
#data_save_dir = "/nfs/pyrex/raid6/hzhang/SmoothPerturbation"
#data_save_dir = "/nfs/pyrex/raid6/hzhang/number_one/data"
i=0
for i_p in range(1):
    #para_str="1.0_"+para[i_p]
    para_str="6.0_3.0"
    file_name = "smooth/mnist_BasicCnn_sbim_"+str(100)+para_str+"_adv_x.mat"
    file_path = os.path.join(data_save_dir,file_name)
    data=si.loadmat(file_path)
    adv_x=data['adv_x']
    #file_name = "cifar10/cifar10_BasicCnn_clipl2_"+str((i+1)*100)+"_"+para_str+"_adv_pre.mat"
    #file_path = os.path.join(data_save_dir,file_name)
    #data=si.loadmat(file_path)
    #adv_pre=data['adv_pre']
    for i in range(1,100):
        file_name = "smooth/mnist_BasicCnn_sbim_"+str((i+1)*100)+para_str+"_adv_x.mat"
        file_path = os.path.join(data_save_dir,file_name)
        data=si.loadmat(file_path)
        adv=data['adv_x']
        adv_x=np.concatenate((adv_x,adv))
        #file_name = "cifar10/cifar10_BasicCnn_clipl2_"+str((i+1)*100)+"_"+para_str+"_adv_pre.mat"
        #file_path = os.path.join(data_save_dir,file_name)
        #data=si.loadmat(file_path)
        #adv=data['adv_pre']
        #adv_pre=np.concatenate((adv_pre,adv))
        
    print(adv_x.shape)
    file_name = "mnist_BasicCnn_sbim_"+para_str+"_adv_x.mat"
    file_path = os.path.join(data_save_dir,file_name)
    si.savemat(file_path,{'adv_x':adv_x})
    #file_name = "cifar10_BasicCnn_clipl2_"+para_str+"_adv_pre.mat"
    #file_path = os.path.join(data_save_dir,file_name)
    #si.savemat(file_path,{'adv_pre':adv_pre})

