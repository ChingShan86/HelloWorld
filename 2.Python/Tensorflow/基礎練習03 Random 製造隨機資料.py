# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

"""
有無placeholder(佔位符)的差別
"""

""" stddev : Standard Deviation 標準差  """
""" b 為 1*2 2D陣列 的框架"""
b = tf.Variable(tf.random_normal([1,2],stddev=1))
""" 補充: b2 為 100 1D陣列 的框架 內容為0~1的數字"""
b2 = tf.Variable(tf.random_uniform([100],0.0,1.0))


""" x1 使用佔位符製造一個空間及格式的空框架"""
x1 = tf.placeholder(tf.float32,shape = (1,2))
x2 = tf.constant([0.7,0.9])

a1 = x1 + b
a2 = x2 + b

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    """ 餵 x1 框架資料 資料為 [[0.7,0.9]]"""
    y1 = sess.run(a1,feed_dict = {x1:[[0.7,0.9]]})
    y2 = sess.run(a2)
    print(sess.run(b))
    print(sess.run(b2))
    print(y1)
    print(y2)


