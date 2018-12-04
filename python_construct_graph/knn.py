from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import glob
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as ss

def similarity(x_img,lamuda):
    """
    [image_feature, image_index] = similarity(x_img,lamuda)
    create the knn lists and feature based on the 4-neighboors pixels
    x_img: M*M images
    lamuda: parameter to control how much the neighboors pixels influence the center pixel
    image_feature: the k*N list of corresponding similarities for knn
    image_index: k*N list of knn per vector (used to build the knn graph)
    """
    shape = x_img.shape()
    im_m = x_img
    im_mod = np.pad(np.array(x_img,dtype=float), ((1,1),(1,1)),'constant',constant_values=np.inf)
    im_r = im_mod[0:shape[0]  ,1:shape[1]+1]
    im_l = im_mod[2:shape[0]+2,1:shape[1]+1]
    im_u = im_mod[1:shape[0]+1,0:shape[1]]
    im_d = im_mod[1:shape[0]+1,2:shape[1]+2]
    im_r = np.exp(-lamuda*np.square(im_r-im_m))
    im_l = np.exp(-lamuda*np.square(im_l-im_m))
    im_u = np.exp(-lamuda*np.square(im_u-im_m))
    im_d = np.exp(-lamuda*np.square(im_d-im_m))
    im_r = np.reshape(im_r,(1,shape[0]*shape[1]))
    im_l = np.reshape(im_l,(1,shape[0]*shape[1]))
    im_u = np.reshape(im_u,(1,shape[0]*shape[1]))
    im_d = np.reshape(im_d,(1,shape[0]*shape[1]))
    image_feature = np.concatenate((im_r,im_l,im_u,im_d),axis=0)

    image_index = np.zeros((4,shape[0]*shape[1]+1))
    ind = np.arange(shape[0]*shape[1]).reshape(shape[0],shape[1])
    ind_mod = np.pad(ind,((1,1),(1,1)),'constant',constant_values = shape[0]*shape[1]+1)
    ind_r = ind_mod[0:shape[0]  ,1:shape[1]+1].reshape(1,shape[0]*shape[1])
    ind_l = ind_mod[2:shape[0]+2,1:shape[1]+1].reshape(1,shape[0]*shape[1])
    ind_u = ind_mod[1:shape[0]+1,0:shape[1]].reshape(1,shape[0]*shape[1])
    ind_d = ind_mod[1:shape[0]+1,2:shape[1]+2].reshape(1,shape[0]*shape[1])
    image_index[:,0:shape[0]*shape[1]] = np.concatenate((ind_r,ind_l,ind_u,ind_d),axis=0)

    return image_feature, image_index

def knngraph(knn_ind,sim_fea):
    """
    A = knngraph(knn_ind, sim_fea)
    create the affinity matrix for the mutual kNN graph based on the knn lists
    knn_ind: kxN list of knn per vector
    sim_fea: kxN list of corresponding similarities for knn
    A: sparse affinity matrix NxN
    """
    shape = knn_ind.shape()
    sim_fea[sim_fea<0] = 0 # similarity should be non-negative

    data = np.array([],dtype=float)
    row  = np.array([],dtype=int)
    col  = np.array([],dtype=int)
    for i in range(shape[1]):
        tmp = knn_ind[:,i:i+1]
        mem = (tmp<=shape[1]*shape[0])
        nk  = np.sum(mem+0)
        if nk >0:
            row = np.append(row,i*np.ones((nk,)))
            col = np.append(col,tmp[mem])
            tp_v = sim_fea[:,i:i+1]
            data = np.append(data,tp_v[mem])
    A=csr_matrix((data,(row,col)),shape=(shape[0]*shape[1],shape[0]*shape[1]))
    A.setdiag(0) # diagonal to 0
    return A

def transition_matrix(W):
    """
    S =  transition_matrix(W)
    calculate the transition matrix of W
    W: N*N sparse affinity matrix
    S: N*N sparse transition matrix
    """
    shape = W.shape()
    tmp = W.sum(1)
    tmp = np.power(tmp,-0.5)
    D = ss.spdiags(tmp.transpose(),np.array([0]),shape[0],shape[1])
    S = D * W * D
    return S






