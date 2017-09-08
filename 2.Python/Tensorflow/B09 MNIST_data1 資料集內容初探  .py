# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 22:54:03 2017

@author: ching shan
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("data/MNIST_data/",one_hot=True)

""" 來看看 mnist 的資料格式及數量 """
print(type(mnist))
print(mnist.train.num_examples )
print(mnist.validation.num_examples)
print(mnist.test.num_examples)

print("---")

print("MNIST訓練還有測試的資料集長得如何呢?")
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels
print("train img type: ",type(train_img)) # 55000 rows 784 cols (28*28 pixels**2)
print("train img dimension: ",train_img.shape)
print("train label type: ",type(train_label))
print("train label dimension: ",train_img.shape)
print(" test_img  type : ",type(test_img))
print(" test_img dimension : ",test_img.shape)
print(" test_label type : ",type(test_label))
print(" test_label dimension : " ,test_label.shape) # 10000 rows 10 cols (0~9的輸出)
# pixel點是介於 0 ~ 1 的數值

print("---")

# 印出用於訓練庫中的圖案 的第一筆資料 數字7 的 image像素檔案資料 及 他的 one hot label 資料
print(np.array(train_img[0])) # 1D資料
print("---")
print(np.reshape(train_img[0],(28,28))) # 2D 資料
print(train_label[0])

print("---")

print("實際印出MNIST的資料集看看是長怎樣")
trainimg = mnist.train.images
trainlabel = mnist.train.labels
nsample = 1
randidx = np.random.randint(trainimg.shape[0],size = nsample) 
print(randidx)
#for i in [0,1,54998,54999]: #共有55000筆資料 0~54999
#    curr_img = np.reshape(trainimg[i,:],(28,28)) #將一長串的row資料變成28*28的矩形
#    curr_label = np.argmax(trainlabel[i,:]) #輸出判斷後機率最大的數字
#    plt.matshow(curr_img,cmap = plt.get_cmap('Blues'))
#    print("第"+str(i+1)+" row 的 training data, Label is "+str(curr_label))
print(mnist.test.images.shape)

batch_x,batch_y = mnist.train.next_batch(100)
print(batch_x)
print(batch_x.shape)
print(type(batch_x))