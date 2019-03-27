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
from basic_cnn_models import make_basic_cnn, make_simple_cnn

def load_test_data():
    data_save_dir = "/nfs/pyrex/raid6/hzhang/SmoothPerturbation"
    file_name = "X_test.mat"
    load_path = os.path.join(data_save_dir,file_name)
    data=si.loadmat(load_path)
    X_test = data['X_test']
    file_name = "Y_test.mat"
    load_path = os.path.join(data_save_dir,file_name)
    data=si.loadmat(load_path)
    Y_test = data['Y_test']
    file_name = "P_test.mat"
    load_path = os.path.join(data_save_dir,file_name)
    data=si.loadmat(load_path)
    P_test = data['P_test']
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)
    return X_test, Y_test, P_test

def load_adv_data(name):
    data_save_dir = "/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/update"
    #data_save_dir = "/nfs/pyrex/raid6/hzhang/SmoothPerturbation"
    file_name= "mnist_BasicCnn_" + name + "_adv_x.mat"
    #file_name= "mnist_AdvTrain_" + name + "_adv_x.mat"
    file_path = os.path.join(data_save_dir,file_name)
    data=si.loadmat(file_path)
    X_adv=data['adv_x']
    X_adv=np.reshape(X_adv,(10000,28,28,1))
    #print('X_adv shape:', X_adv.shape)
    return X_adv

###########
## create TF session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
## Define the input TF placeholder
x_ = tf.placeholder(tf.float32, shape=(None, 28, 28,1)) # for mnist
y_ = tf.placeholder(tf.float32, shape=(None,10)) # for mnist
model = make_basic_cnn()
preds = model(x_ )
tf_model_load(sess,'models/basic_cnn_adv.ckpt')
#tf_model_load(sess,'/udd/hzhang/SmoothPerturbation/compare/models/basic_cnn_fgsm_0.3.ckpt')


def result(X_test,Y_test,P_test,X_adv):
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x_, y_, preds, X_adv, Y_test, args=eval_params)
    accuracy_o = model_eval(sess, x_, y_, preds, X_test, Y_test, args=eval_params)
    grey = np.zeros((10000,28,28,1))*0.5
    L2_worst = np.sum((grey-X_test)**2,axis=(1,2,3))**.5
    L2_norm = np.sum((X_adv-X_test)**2,axis=(1,2,3))**.5
    Y_adv = sess.run(preds,feed_dict={x_:X_adv,y_:Y_test})
    ## change mind rate
    ori_y = np.argmax(Y_test,axis=1)
    pre_y = np.argmax(Y_adv,axis=1)
    dif = np.abs(ori_y-pre_y)
    ind = np.ones(ori_y.shape)
    ind[np.nonzero(dif)] = 0

    p_test = sess.run(preds,feed_dict={x_:X_test,y_:Y_test})
    p_y = np.argmax(p_test,axis=1)
    dif = np.abs(ori_y-p_y)
    ind_o = np.ones(ori_y.shape)
    ind_o[np.nonzero(dif)] = 0

    dif = np.abs(pre_y-p_y)
    ind_c = np.ones(p_y.shape)
    ind_c[np.nonzero(dif)] = 0

    return accuracy,L2_norm,ind,ind_o,ind_c,L2_worst,accuracy_o

X_test, Y_test, P_test = load_test_data()
eval_params ={'batch_size':128}

save_dir = "/nfs/nas4/data-hanwei/data-hanwei/DATA/SmoothPerturbation/update/whole_image"
#save_dir = "/nfs/pyrex/raid6/hzhang/SmoothPerturbation/step_iter_smooth/whole_image/"

## Carlini Wagner

#name_list={"fgsm_0.2","fgsm_0.1","fgsm_0.4","bim_0.1_0.08","fgsm_0.5","bim_0.2_0.08","pgd_2.0_3.0","pgd_4.0_3.0","pgd_6.0_3.0","pgd_1.0_3.0","bim_0.4_0.08","pgd_3.0_3.0","bim_0.5_0.08"}
#name_list = {"fgsm_0.3","bim_0.3_0.08","cw","pgd_2.0_0.3"}
name_list ={"sbim_1.0_3.0","sbim_2.0_3.0","sbim_3.0_3.0","sbim_4.0_3.0","sbim_5.0_3.0","sbim_6.0_3.0"}
#name_list ={"bim_0.03_0.08","bim_0.05_0.08","bim_0.0625_0.08","bim_0.07_0.08","bim_0.09_0.08","bim_0.15_0.08","bim_0.18_0.08","bim_1.0_3.0","bim_1.5_3.0","bim_1.75_3.0","bim_2.25_3.0","bim_2.5_3.0","bim_3.0_3.0","bim_5.0_3.0","fgsm_0.04","fgsm_0.07","fgsm_0.09","fgsm_0.15","fgsm_0.18","sbim_1.5_3","sbim_1.75_3","sbim_2.5_3","sbim_3.25_3","sbim_3_3","sbim_5_3"}
#name_list = {"bim_0.1_0.08","fgsm_0.1","cw_1.0_15.0","clipl2_1.0_15.0"}
for name in name_list: 
    X_adv = load_adv_data(name)
    acc, l2_n, ind, ind_o,ind_c,L2_w, acc_o = result(X_test,Y_test,P_test,X_adv)
    
    file_name = "mnist_BasicCnn_"+name+".mat"
    #file_name = "trasf_mnist_BasicCnn_"+name+".mat"
    #file_name = "adv_mnist_BasicCnn_"+name+".mat"
    save_path = os.path.join(save_dir,file_name)
    si.savemat(save_path,{'acc':acc,'l2':l2_n,'p':ind,'ori_a':ind_o,'c':ind_c,'l2_worst':L2_w,'acc_o':acc_o})

