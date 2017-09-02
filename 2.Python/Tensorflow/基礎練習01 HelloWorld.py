# -*- coding: utf-8 -*-
"""
使用tensorflow列印出 Hello, World!
"""

import numpy as np
import tensorflow as tf

hello = tf.constant("Hello, World!")
""" 方法一 (只會給出結構) """
#print(hello)

""" 方法二 """
#sess = tf.Session()
#print(sess.run(hello))

""" 方法三 """
with tf.Session() as sess:
    print(sess.run(hello))
