# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf

""" 
常數/變數 取名字 
"""
a = tf.constant([1,2],name ='apple')
b = tf.constant([3,2],name = 'banana')
""" 測試1 框架資訊的Const會被改成自己所取的名字apple"""
print(a)
# Tensor("apple_2:0", shape=(2,), dtype=int32)
