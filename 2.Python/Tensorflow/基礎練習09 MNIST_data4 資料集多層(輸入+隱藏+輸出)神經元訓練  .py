# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as input_dataset
mnist = input_dataset.read_data_sets("data/MNIST_data/",one_hot = True)

def add_layer(input_data,in_size,out_size,activate_f = None):
    W = tf.Variable(tf.zeros([in_size,out_size])+0.1)
    b = tf.Variable(tf.zeros([1,out_size])+0.1)
    prediction = tf.matmul(input_data,W) + b
    if activate_f == None :
        output_data = prediction
    else :
        output_data = activate_f(prediction)
    return output_data

def caculate_loss_methods(y_real_data,y_predict,caculate_loss_type = 1):
    return{ # 建立一個字典 (有如 C++ 的 switch-case)
            1 : tf.reduce_mean(tf.reduce_sum(tf.square(y_real_data-y_predict),reduction_indices=[1])),
            2 : tf.reduce_mean(-tf.reduce_sum(y_real_data*tf.log(y_predict),reduction_indices=[1])),
    }[caculate_loss_type]

def optimizers_methods(loss,optimizer_type = 1):
    return {
            1 : tf.train.GradientDescentOptimizer(0.5).minimize(loss), # 學習速率 0.5
            2 : tf.train.AdamOptimizer(0.005).minimize(loss), # 學習速率 0.005 (預設建議0.0001)
           # 3 : tf.train.MomentumOptimizer(0.01).minimize(loss)
            4 : tf.train.RMSPropOptimizer(0.003).minimize(loss)
    }[optimizer_type]



x_data = tf.placeholder(tf.float32,[None,784])
y_real_data = tf.placeholder(tf.float32,[None,10])
#y_predict_data = tf.placeholder(tf.float32,size = [None,10])

y_predict = add_layer(x_data,784,10,activate_f = tf.nn.softmax)

loss = caculate_loss_methods(y_real_data,y_predict,1)
train = optimizers_methods(loss,optimizer_type = 1)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100) #每運行一次更新一次batch的100筆資料
        sess.run(train,feed_dict = {x_data:batch_x,y_real_data:batch_y})
        if step % 50 ==0:
            """ 計算出準確度的過程可以寫成def函式 """
            # vvv 餵test資料並計算，使用自己的神經網路計算(預測)出的 y_prediction
            y_prediction = sess.run(y_predict,feed_dict={x_data:mnist.test.images}) 
            # vvv 定義框架 :預測值與真實質的相似程度(相同的那一格儲存True)
            similiar_y_level = tf.equal(tf.arg_max(y_prediction,1),tf.arg_max(mnist.test.labels,1)) 
            # vvv 定義框架:計算True佔所有的百分比
            percent_accuracy = tf.reduce_mean(tf.cast(similiar_y_level,tf.float32)) 
            # vvv 餵資料給框架: 計算True占所有的比例(準確度)
            accuracy = sess.run(percent_accuracy,feed_dict={y_real_data:mnist.test.labels})
            print("acc:",accuracy)
            #print("loss:",sess.run(loss,feed_dict={x_data:mnist.test.images,y_real_data:mnist.test.labels}))
    
