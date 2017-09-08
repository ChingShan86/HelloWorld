# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 13:38:45 2017

@author: ching shan
"""

from __future__ import print_function
from PIL import Image
import tensorflow as tf
import numpy as np
import os

""" 將圖片資料讀入 """
def get_image_paths(img_dir):
    filenames = os.listdir(img_dir)
    filenames = [os.path.join(img_dir, item) for item in filenames]
    return filenames
#pos_filenames和neg_filenames分别对应图片和正常图片的文件名
pos_filenames = get_image_paths("./my_hands/fist")
neg_filenames = get_image_paths("./my_hands/palm")
print("num of pos samples is %d" % len(pos_filenames))
print("num of neg samples is %d" % len(neg_filenames))

print("---")

""" 將資料分成 8:2 等分 """
TRAIN_SEC, TEST_SEC = 0.8, 0.2
pos_sample_num, neg_sample_num = len(pos_filenames), len(neg_filenames)

np.random.shuffle(np.arange(len(pos_filenames)))
np.random.shuffle(np.arange(len(neg_filenames)))

pos_train, pos_test = pos_filenames[: int(pos_sample_num * TRAIN_SEC)], pos_filenames[int(pos_sample_num * TRAIN_SEC) :]
neg_train, neg_test = neg_filenames[: int(neg_sample_num * TRAIN_SEC)], neg_filenames[int(neg_sample_num * TRAIN_SEC) :]

print("Pos sample : train num is %d, test num is %d" % (len(pos_train), len(pos_test)))
print("Neg sample : train num is %d, test num is %d" % (len(neg_train), len(neg_test)))

print("---")

""" 將訓練資料和測試資料分開 """
all_train, all_test = [], []
all_train.extend(pos_train)
all_train.extend(neg_train)
all_test.extend(pos_test)
all_test.extend(neg_test)
all_train_label, all_test_label = np.ones(len(pos_train), dtype=np.int64), np.ones(len(pos_test), dtype=np.int64)
all_train_label.resize(len(all_train))
all_test_label.resize(len(all_test))
print("train num is %d, test num is %d" % (len(all_train), len(all_test)))

print("---")

""" 將資料轉為tfrecords格式 """
def save_as_tfrecord(samples, labels, folder,names):
    classes={'fist','palm'} #人为 设定 2 类
    writer= tf.python_io.TFRecordWriter(names) #要生成的文件
    
    for index,name in enumerate(classes):
        class_path = folder + name + '/'
        for img_name in os.listdir(class_path): 
            img_path=class_path+img_name #每一个图片的地址
    
            img=Image.open(img_path)
            img= img.resize((128,128))
            img_raw=img.tobytes()#将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  #序列化为字符串
    
    writer.close()
    
save_as_tfrecord(all_test, all_test_label, "./my_hands/","test.bin")
save_as_tfrecord(all_train, all_train_label, "./my_hands/","train.bin")

print("---")









