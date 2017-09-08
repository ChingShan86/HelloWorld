import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    layer_name = "layer" + str(n_layer)
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name = "W")
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name = "b")
        with tf.name_scope("WX_plus_b"):
            WX_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = WX_plus_b
        else:
            outputs = activation_function(WX_plus_b)
    tf.summary.histogram(layer_name + "/Weights",Weights) #會於histogram欄位顯示此圖表
    tf.summary.histogram(layer_name + "/biases",biases)
    tf.summary.histogram(layer_name + "/outputs",outputs)
    return outputs

x_datas = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_datas.shape)
y_datas = np.square(x_datas) - 0.5 + noise

with tf.name_scope("Input_Data"): 
    x_data = tf.placeholder(tf.float32, [None, 1],name = 'x_data')
    y_real_data = tf.placeholder(tf.float32, [None, 1],name = 'y_real_data')

l1 = add_layer(x_data, 1, 10, n_layer = 1, activation_function=tf.nn.relu)
y_prediction = add_layer(l1, 10, 1, n_layer = 2 , activation_function=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_real_data - y_prediction),reduction_indices=[1]),name = "loss")
tf.summary.scalar("loss_curve",loss)
with tf.name_scope("train"):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("C:/board/",sess.graph)
    tf.global_variables_initializer().run()

    for i in range(1000):
        # training
        sess.run(train, feed_dict={x_data: x_datas, y_real_data: y_datas})
        if i % 50 == 0:
            result = sess.run(merged,feed_dict={x_data: x_datas, y_real_data: y_datas})
            writer.add_summary(result, i)

