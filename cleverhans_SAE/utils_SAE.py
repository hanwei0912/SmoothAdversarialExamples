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
from . import utils_tf
from . import utils


def multi_sparse(mod, A, batch_size, shape):
    im_mod = tf.reshape(mod, (batch_size, shape[1], shape[2],shape[3]))
    im_mod = tf.pad(im_mod, paddings =[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT")
    im_r = tf.slice(im_mod, [0, 0, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
    im_l = tf.slice(im_mod, [0, 2, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
    im_u = tf.slice(im_mod, [0, 1, 0, 0], [batch_size, shape[1], shape[2], shape[3]])
    im_d = tf.slice(im_mod, [0, 1, 2, 0], [batch_size, shape[1], shape[2], shape[3]])
    im_r = tf.reshape(im_r, (batch_size, 1, shape[1]*shape[2], shape[3]))
    im_l = tf.reshape(im_l, (batch_size, 1, shape[1]*shape[2], shape[3]))
    im_u = tf.reshape(im_u, (batch_size, 1, shape[1]*shape[2], shape[3]))
    im_d = tf.reshape(im_d, (batch_size, 1, shape[1]*shape[2], shape[3]))
    im_a = tf.concat([im_r, im_l, im_u, im_d], 1)
    A = tf.reshape(A, (batch_size, 4, shape[1]*shape[2], shape[3]))
    smo = tf.multiply(A, im_a)
    smo = tf.reduce_sum(smo, axis=1)
    smo = tf.reshape(smo, (batch_size, shape[1]*shape[2], shape[3]))
    smo_t = tf.add(smo, mod)
    return smo_t

def CG_loop(i,r,p,s,x,A,batch_size,shape):
    q = multi_sparse(p, A, batch_size, shape)
    alpha_t = tf.div(s, tf.matmul(tf.transpose(p, [0, 2, 1]),q))
    alpha_t = tf.map_fn(lambda x: tf.diag_part(x),alpha_t,dtype=tf.float32)
    alpha =tf.tile(tf.reshape(alpha_t,[batch_size,1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
    x = tf.add(x, tf.multiply(alpha, p))
    r = tf.subtract(r, tf.multiply(alpha, q))
    t = tf.matmul(tf.transpose(r, [0, 2, 1]),r)
    beta_t = tf.div(t, s)
    beta_t = tf.map_fn(lambda x: tf.diag_part(x),beta_t,dtype=tf.float32)
    beta = tf.tile(tf.reshape(beta_t,[batch_size,1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
    p = tf.add(r, tf.multiply(beta, p))
    s = t
    i = tf.add(i, 1)
    return i, r, p, s, x


def CG(A, modifier,batch_size,shape):
    def while_condition(i, r, p, s, x): return tf.less(i, 50)
    def body(i,r,p,s,x):
        i,r,p,s,x = CG_loop(i,r,p,s,x,A,batch_size,shape)
        return i,r,p,s,x
    global i
    i = tf.constant(0)
    r = modifier - multi_sparse(modifier, A, batch_size, shape)
    p = r
    s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
    x = modifier
    i, r, p, s, smo_mod = tf.while_loop(
        while_condition, body, [i, r, p, s, x])
    ## norm
    z_d = tf.ones_like(modifier)
    i = tf.constant(0)
    r = z_d - multi_sparse(z_d, A, batch_size, shape)
    p = r
    s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
    x = z_d
    i, r, p, s, div_z = tf.while_loop(
        while_condition, body, [i, r, p, s, x])

    smo_mod = tf.div(smo_mod, div_z)
    smo_mod = tf.reshape(smo_mod, shape)
    return smo_mod
