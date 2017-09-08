# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

"""
一般數字 四則運算
"""
""" 建立 1D陣列 """
a = tf.constant([1,2,3,4])
b = tf.constant([5,6,7,8])

result1 = a + b
result2 = a * b
result3 = a % b
result4 = a // b

with tf.Session() as sess:
    print(sess.run(result1))
    print(sess.run(result2))
    print(sess.run(result3))
    print(sess.run(result4))
#[ 6  8 10 12]
#[ 5 12 21 32]
#[1 2 3 4]
#[0 0 0 0]
    
"""
矩陣 乘法
"""
""" 建立2D陣列 """
A = tf.constant([[1,2],
                 [3,4]])
B = tf.constant([[5,6],
                 [7,8]])
    
#R1 = tf.Variable(A*B) <--- error
#R2 = tf.Variable(tf.matmul(B,A))  <--- error
R3 = tf.matmul(A,B)
R4 = tf.matmul(B,A)

with tf.Session() as sess:
    print(sess.run(A))
    print(sess.run(B))
    print(sess.run(R3),"!=",sess.run(R4))
    
    

