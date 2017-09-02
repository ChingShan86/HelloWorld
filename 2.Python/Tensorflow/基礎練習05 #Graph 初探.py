# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

""" **** 用於tensor board使用 **** """

a = tf.constant(2, tf.int16)
b = tf.constant(4, tf.float32)

graph = tf.Graph()
with graph.as_default():
    a = tf.Variable(8, tf.float32)
    b = tf.Variable(tf.zeros([2,2], tf.float32))
    
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(session.run(a))
    print(session.run(b))