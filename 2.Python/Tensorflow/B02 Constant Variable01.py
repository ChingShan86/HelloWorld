# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

""" 
常數 完全不能更改
"""
a = tf.constant(2,tf.int16)
b = tf.constant(4,tf.float32)

""" 測試1 """
print(a,b)
# Tensor("Const_10:0", shape=(), dtype=int16) """
# Tensor("Const_11:0", shape=(), dtype=float32) """

""" 測試2 """
with tf.Session() as sess:
    print(sess.run(a),sess.run(b))
# 2 4.0 

#########################################

"""
變數 可被不斷更改
"""
d = tf.Variable(2,tf.int16)
e = tf.Variable(2,tf.float32)

""" 測試1 """
print(d,e)
# <tf.Variable 'Variable:0' shape=() dtype=int32_ref> 
# <tf.Variable 'Variable_1:0' shape=() dtype=int32_ref>

""" 測試2 """
with tf.Session() as sess:
    tf.global_variables_initializer().run() # <-- 變數必加上
    print(sess.run(d),sess.run(e))
# 2 2  
    
#########################################
    
"""
常數 
"""
""" 1. 以np方式建立2D陣列 """
g1 = tf.constant(np.zeros(shape=(2,3),dtype=np.float32))
with tf.Session() as sess:
    print(sess.run(g1))
# [[0. 0. 0.]
#  [0. 0. 0.]]
    
""" 2. 以內建的tf方式建立2D陣列 """
g2 = tf.zeros([2,3],tf.float64)
with tf.Session() as sess:
    print(sess.run(g2))
# [[0. 0. 0.]
#  [0. 0. 0.]]
    
""" 3. 1D陣列 """
g3 = tf.zeros([8],tf.float64)
with tf.Session() as sess:
    print(sess.run(g3))
# [ 0.  0.  0.  0.  0.  0.  0.  0. ]
    
""" 4. 3*(2*3) 3D陣列"""
g4 = tf.zeros([3,2,3], tf.float64)
with tf.Session() as sess:
    print(sess.run(g4))
#[[[ 0.  0.  0.]
#  [ 0.  0.  0.]]

# [[ 0.  0.  0.]
#  [ 0.  0.  0.]]

# [[ 0.  0.  0.]
#  [ 0.  0.  0.]]] 

""" 6D陣列 ..."""
g5 = tf.zeros([2,3,2,1,2,2],tf.int16)
with tf.Session() as sess:
    print(sess.run(g5))   
    
#########################################

"""
變數 以tf方式 建立 3個2*2的 3D陣列 內容皆為1
"""
i = tf.Variable(tf.ones([3,2,2],tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run() # remember plz
    print(sess.run(i))
# [[[1. 1.]]  [[1. 1.]]  [[1. 1.]]
#   [1. 1.]]   [1. 1.]]   [1. 1.]]]


