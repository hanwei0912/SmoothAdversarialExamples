from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings
import pdb
import gc
from cleverhans import utils_tf
from cleverhans import utils


def multi_sparse(mod, A, shape):
    im_mod = tf.pad(im_mod, paddings =[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT")
    im_r = tf.slice(im_mod, [0, 0, 1, 0], [shape[0], shape[1], shape[2], shape[3]])
    im_l = tf.slice(im_mod, [0, 2, 1, 0], [shape[0], shape[1], shape[2], shape[3]])
    im_u = tf.slice(im_mod, [0, 1, 0, 0], [shape[0], shape[1], shape[2], shape[3]])
    im_d = tf.slice(im_mod, [0, 1, 2, 0], [shape[0], shape[1], shape[2], shape[3]])
    smo = tf.add(tf.multiply(A[:,0,:,:,:],im_r),mod)
    smo = tf.add(tf.multiply(A[:,1,:,:,:],im_l),smo)
    smo = tf.add(tf.multiply(A[:,2,:,:,:],im_u),smo)
    smo = tf.add(tf.multiply(A[:,3,:,:,:],im_d),smo)
    return smo

def CG_loop(i, r, p, s, x, A, shape):
    q = multi_sparse(p, A, shape)
    pq = tf.reduce_sum(tf.multiply(p,q),axis=[1,2])
    alpha = tf.div(s,pq)
    alpha =tf.tile(tf.reshape(alpha,[shape[0],1,1,shape[3]]),multiples=[1,shape[1],shape[2],1])
    x = tf.add(x, tf.multiply(alpha, p))
    r = tf.subtract(r, tf.multiply(alpha, q))
    t = tf.reduce_sum(tf.multiply(r,r),axis=[1,2])
    beta = tf.div(t, s)
    beta = tf.tile(tf.reshape(beta,[shape[0],1,1,shape[3]]),multiples=[1,shape[1],shape[2],1])
    p = tf.add(r, tf.multiply(beta, p))
    s = t
    i = tf.add(i, 1)
    return i, r, p, s, x


def CG(A, modifier,shape):
    def while_condition(i, r, p, s, x): return tf.less(i, 50)
    def body(i,r,p,s,x):
        i,r,p,s,x = CG_loop(i,r,p,s,x,A,shape)
        return i,r,p,s,x
    global i
    i = tf.constant(0)
    r = modifier - multi_sparse(modifier, A, shape)
    p = r
    s = tf.reduce_sum(tf.multiply(r,r),axis=[1,2])
    x = modifier
    i, r, p, s, smo_mod = tf.while_loop(
        while_condition, body, [i, r, p, s, x])
    return smo_mod

def Norm_CG(smo_mod, div_z):
    return tf.div(smo_mod,div_z)
