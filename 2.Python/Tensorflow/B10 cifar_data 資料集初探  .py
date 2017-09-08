# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import os


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def one_hot_encode(np_array):
    return (np.arange(10) == np_array[:,None]).astype(np.float32)

def reformat_data(dataset, labels, image_width, image_height, image_depth):
    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
    np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels

cifar10_folder = './data/CIFAR10_data/cifar-10-batches-py/'
train_datasets = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', ]
test_dataset = ['test_batch']
c10_image_height = 32
c10_image_width = 32
c10_image_depth = 3
c10_num_labels = 10
c10_image_size = 32 #Ahmet Taspinar的代码缺少了这一语句

with open(cifar10_folder + test_dataset[0], 'rb') as f0:
    c10_test_dict = pickle.load(f0, encoding='bytes')

c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
test_dataset_cifar10, test_labels_cifar10 = reformat_data(c10_test_dataset, c10_test_labels, c10_image_size, c10_image_size, c10_image_depth)

c10_train_dataset, c10_train_labels = [], []
for train_dataset in train_datasets:
    with open(cifar10_folder + train_dataset, 'rb') as f0:
        c10_train_dict = pickle.load(f0, encoding='bytes')
        c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']
 
        c10_train_dataset.append(c10_train_dataset_)
        c10_train_labels += c10_train_labels_

c10_train_dataset = np.concatenate(c10_train_dataset, axis=0) # concatenate 串聯
train_dataset_cifar10, train_labels_cifar10 = reformat_data(c10_train_dataset, c10_train_labels, c10_image_size, c10_image_size, c10_image_depth)
del c10_train_dataset
del c10_train_labels

print("訓練及所包含的標籤: {}".format(np.unique(c10_train_dict[b'labels'])))
print('訓練集的資料维度(dimention)', train_dataset_cifar10.shape, train_labels_cifar10.shape)
print('測試用的資料维度(dimention)', test_dataset_cifar10.shape, test_labels_cifar10.shape) # test_dataset 10000列(筆圖片)、32行寬、32行高、深度3(BGR) 4維資料 / test_labels 10000列 10行(類別) 2維資料
print('測試用的資料型態(type)',type(test_dataset_cifar10),type(test_labels_cifar10)) # np.array格式
print(test_dataset_cifar10[0]) # 第一筆所有資料 圖寬 圖高 及顏色BGR
print("---")
print(test_dataset_cifar10[0][0])
print("---")
print(test_dataset_cifar10[0][0][0]) # 印出BGR資料
print(test_dataset_cifar10[0][0][0][0]) # 0000:B 0001:G 0002:R  0 ~ 2

""" 印出來長怎樣 """
for i in [100]: #共有10000筆測試用資料 0~9999
    curr_img = test_dataset_cifar10[i,:,:,0]+test_dataset_cifar10[i,:,:,1]+test_dataset_cifar10[i,:,:,2] #將一長串的row資料變成28*28的矩形
    curr_label = np.argmax(test_labels_cifar10[i,:]) #輸出判斷後機率最大的類別
    plt.matshow(curr_img,cmap = plt.get_cmap('gray'))
    print("第"+str(i+1)+" row 的 training data, Label is "+str(curr_label))

