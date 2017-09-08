import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as input_dataset
mnist = input_dataset.read_data_sets("data/MNIST_data/",one_hot = True)

def add_layer(input_data,in_size,out_size,n_layer,activate_f = None):
    layer_name = "layer" + str(n_layer)
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights"):
            W = tf.Variable(tf.zeros([in_size,out_size])+0.1,name="Weights")
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([1,out_size])+0.1,name="biases")
        prediction = tf.matmul(input_data,W) + b
        if activate_f == None :
            output_data = prediction
        else :
            output_data = activate_f(prediction)
    tf.summary.histogram(layer_name + "/Weights",W) #會於histogram欄位顯示此圖表
    tf.summary.histogram(layer_name + "/biases",b)
    tf.summary.histogram(layer_name + "/outputs",output_data)
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

with tf.name_scope("Input_Data"):
    x_data = tf.placeholder(tf.float32,[None,784], name = "x_data")
    y_real_data = tf.placeholder(tf.float32,[None,10], name = "y_real_data")
    #y_predict_data = tf.placeholder(tf.float32,size = [None,10])

y_predict = add_layer(x_data,784,10,n_layer = 1 ,activate_f = tf.nn.softmax)

with tf.name_scope("Loss"):
    loss = caculate_loss_methods(y_real_data,y_predict,2)
tf.summary.scalar("Loss_Curve",loss)
with tf.name_scope("Train"):
    train = optimizers_methods(loss,optimizer_type = 1)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("C:/board/",sess.graph)
    tf.global_variables_initializer().run()
    for step in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100) #每運行一次更新一次batch的100筆資料
        if step % 50 ==0:
#            y_prediction = sess.run(y_predict,feed_dict={x_data:mnist.test.images}) 
#            similiar_y_level = tf.equal(tf.arg_max(y_prediction,1),tf.arg_max(mnist.test.labels,1)) 
#            percent_accuracy = tf.reduce_mean(tf.cast(similiar_y_level,tf.float32)) 
#            accuracy = sess.run(percent_accuracy,feed_dict={y_real_data:mnist.test.labels}) 
#            tf.summary.scalar("loss_curve",accuracy)
#            print("loss:",sess.run(loss,feed_dict={x_data:mnist.test.images,y_real_data:mnist.test.labels}))
            result = sess.run(merged,feed_dict={x_data: batch_x, y_real_data: batch_y})
            writer.add_summary(result,step)
        else:
            sess.run(train,feed_dict = {x_data:batch_x,y_real_data:batch_y})