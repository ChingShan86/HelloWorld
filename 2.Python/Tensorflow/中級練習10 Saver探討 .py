# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:46:03 2017

@author: ching shan
"""

import tensorflow as tf
import os
#os.makedirs("./tmp/model")
#os.makedirs("./tmp/model-subset")

# 建立一些變數以及對應的名字．

v1 = tf.Variable([0.1, 0.1], name="v1")
v2 = tf.Variable([0.2, 0.2], name="v2")

# 建立所有 variables 的初始化 ops
init_op = tf.global_variables_initializer()

# 建立 saver 物件
saver = tf.train.Saver()

with tf.Session() as sess:
    
    # 執行初始化
    sess.run(init_op)
    
    #重新指定 v2 的值
    ops = tf.assign(v2, [0.3, 0.3])
    sess.run(ops)
    
    print(sess.run(tf.global_variables()))
    # ... 中間略去許多模型定義以及訓練，例如可以是 MNIST 的定義以及訓練
    
    save_path = saver.save(sess, "/tmp/model/model.ckpt") # 儲存模型到 /tmp/model.ckpt