'''
Code for Rapid Treatment Planning for Low-dose-rate Prostate Brachytherapy with TP-GAN

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca
Robotics and Control Laboratory, University of British Columbia, Vancouver, Canada
'''

import keras.backend as K
import tensorflow as tf
from keras import losses
import numpy as np

# Sum of L1 and Adjacent Seed loss
def l1_and_adj_seed_loss(alpha):
    def loss(y_true, y_pred):
        return K.sum(l1_loss(y_true, y_pred)) + alpha * K.sum(adj_seeds_loss(y_true, y_pred))
    return loss

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def adj_seeds_loss(y_true, y_pred, smooth=0.00001):

    kernel1 = [[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]]

    kernel2 = [[0, 1, 0],
               [1, 7, 1],
               [0, 1, 0]]

    kernel = np.concatenate((np.expand_dims(kernel1, axis = -1),np.expand_dims(kernel2, axis = -1),np.expand_dims(kernel1, axis = -1)),axis = -1)
    kernel = tf.expand_dims(kernel, axis=3)
    kernel = tf.expand_dims(kernel, axis=4)

    loss = tf.nn.convolution(tf.cast(y_pred, tf.float32), tf.cast(kernel, tf.float32), strides=None, padding='SAME') # Convolves to finds pixels that doesnt follow adjacency rule

    return K.sum(tf.nn.relu(loss - 5))

def find_prohibited_needles(threshold, y_pred):

    temp = y_pred
    temp = tf.clip_by_value(temp, 0, threshold) # Converting to 0-0.5, anything above 0.5 is 0.5
    temp = tf.where(tf.equal(temp, threshold), tf.ones_like(temp), tf.zeros_like(temp))

    kernel1 = [1, 1, 1],
    kernel1 = tf.expand_dims(kernel1, axis=2)
    kernel1 = tf.expand_dims(kernel1, axis=3)
    pn_1 = tf.nn.convolution(tf.cast(temp, tf.float32), tf.cast(kernel1, tf.float32), strides=None, padding='SAME') # Convolves to finds pixels that doesnt follow adjacency rule
    pn_1 = tf.clip_by_value(pn_1, 0, 3)
    pn_1 = tf.where(tf.equal(pn_1, 3), x=tf.ones_like(pn_1), y=tf.zeros_like(pn_1), name=None)

    kernel2 = [[1], [1], [1], [1]]
    kernel2 = tf.expand_dims(kernel2, axis=2)
    kernel2 = tf.expand_dims(kernel2, axis=3)
    pn_2 = tf.nn.convolution(tf.cast(temp, tf.float32), tf.cast(kernel2, tf.float32), strides=None, padding='SAME') # Convolves to finds pixels that doesnt follow adjacency rule
    pn_2 = tf.clip_by_value(pn_2, 0, 4)
    pn_2 = tf.where(tf.equal(pn_2, 4), x=tf.ones_like(pn_2), y=tf.zeros_like(pn_2), name=None)

    kernel3 = [[1, 0], [1, 1], [1, 0]]
    kernel3 = np.squeeze(kernel3)
    kernel3 = tf.expand_dims(kernel3, axis=2)
    kernel3 = tf.expand_dims(kernel3, axis=3)
    pn_3 = tf.nn.convolution(tf.cast(temp, tf.float32), tf.cast(kernel3, tf.float32), strides=None, padding='SAME') # Convolves to finds pixels that doesnt follow adjacency rule
    pn_3 = tf.clip_by_value(pn_3, 0, 4)
    pn_3 = tf.where(tf.equal(pn_3, 4), x=tf.ones_like(pn_3), y=tf.zeros_like(pn_3), name=None)

    kernel4 = [[1, 1], [1, 1]]
    kernel4 = np.squeeze(kernel4)
    kernel4 = tf.expand_dims(kernel4, axis=2)
    kernel4 = tf.expand_dims(kernel4, axis=3)
    pn_4 = tf.nn.convolution(tf.cast(temp, tf.float32), tf.cast(kernel4, tf.float32), strides=None, padding='SAME') # Convolves to finds pixels that doesnt follow adjacency rule
    pn_4 = tf.clip_by_value(pn_4, 0, 4)
    pn_4 = tf.where(tf.equal(pn_4, 4), x=tf.ones_like(pn_4), y=tf.zeros_like(pn_4), name=None)

    pn = pn_1 + pn_2 + pn_3 + pn_4

    return K.sum(pn, axis=(1, 2))


def find_adjacent_seeds(threshold, y_pred):

    temp = y_pred
    temp = tf.clip_by_value(temp,0,threshold) # Converting to 0-0.5, anything above 0.5 is 0.5
    temp = tf.where(tf.equal(temp, threshold), tf.ones_like(temp), tf.zeros_like(temp))


    kernel1 = [[0, 0, 0],
               [0, 0.5, 0],
               [0, 0, 0]]

    kernel2 = [[0, 0.5, 0],
               [0.5, 4, 0.5],
               [0, 0.5, 0]]

    kernel = np.concatenate((np.expand_dims(kernel1, axis=-1), np.expand_dims(kernel2, axis=-1), np.expand_dims(kernel1, axis=-1)), axis=-1)
    kernel = tf.expand_dims(kernel, axis=3)
    kernel = tf.expand_dims(kernel, axis=4)

    aj_seeds = tf.nn.convolution(tf.cast(temp, tf.float32), tf.cast(kernel, tf.float32), strides=None, padding='SAME') # Convolves to finds pixels that doesnt follow adjacency rule

    aj_seeds = tf.clip_by_value(aj_seeds, 0, 4.1)
    aj_seeds = tf.where(tf.equal(aj_seeds, 4.1), x=tf.ones_like(aj_seeds), y=tf.zeros_like(aj_seeds), name=None)

    return K.sum(aj_seeds, axis=(1, 2, 3))






