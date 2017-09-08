# -*- coding: utf-8 -*-
"""
使用tensorflow列印出 Hello, World!
"""

import numpy as np
import tensorflow as tf

hello = tf.constant("Hello, World!")
""" 方法1 只會印出結構框架 """
## Tensor("Const_1:0", shape=(), dtype=string)
#print(hello)

""" 方法2  """
## b'Hello, World!'
#sess = tf.Session()
#print(sess.run(hello))

""" 方法3  """
## b'Hello, World!'
with tf.Session() as sess:
    print(sess.run(hello))
