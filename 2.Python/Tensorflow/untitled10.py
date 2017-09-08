# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:13:28 2017

@author: ching shan
"""

import numpy as np
import tensorflow as tf


def labels_one_hot(labels):
    label_one_hot = tf.one_hot(labels,depth=10,on_value=1.0,off_value=0.0,axis = -1)
    return label_one_hot

ys1 = tf.placeholder(tf.uint8, [None])
print(ys1)
y_one_hot = labels_one_hot(ys1)
print(y_one_hot)

print("----")

ys2 = tf.placeholder(tf.float32, [None,10])
print(ys2)

print("---")

xs = tf.placeholder(tf.float32, [None,3072],name = "x_data")
print(xs)
x_image = tf.reshape(xs, [-1, 32, 32, 3])
print(x_image)

def one_hot_encode(np_array):
    return (np.arange(10) == np_array[:,None]).astype(np.float32)
s = [6,7]
b = np.array(s)
print(b)
#a = np.arange(10)
#print(a)
print(b[:,None])
c = one_hot_encode(b)
print(c)