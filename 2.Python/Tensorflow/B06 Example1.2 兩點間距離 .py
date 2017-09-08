# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

point1 = tf.constant([1,2],tf.float32)
point2 = tf.constant([4,6],tf.float32)

""" 若計算出的距離有小數點 一開始可點可定義為float型態"""
diff = tf.subtract(point1,point2)
power = tf.pow(diff,tf.constant(2.0, shape=(1,2)))
add = tf.reduce_sum(power) # <--- important to learn
distance = tf.sqrt(add)
# d = tf.square(add) square為平方 sqrt為開方 

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(point1),sess.run(point1))
    print(sess.run(diff))
    print(sess.run(power))
    print(sess.run(add))
    print(sess.run(distance))