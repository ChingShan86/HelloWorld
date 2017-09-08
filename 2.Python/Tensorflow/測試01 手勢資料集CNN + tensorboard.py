import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as input_dataset
mnist = input_dataset.read_data_sets("data/MNIST_data/",one_hot = True)


def weight_variable(shape):
    with tf.name_scope("Weights"):
        initial = tf.truncated_normal(shape, stddev=0.1,name="Weights")
    return tf.Variable(initial)

def bias_variable(shape):
    with tf.name_scope("biases"):
        initial = tf.constant(0.1, shape=shape,name="biases")
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
with tf.name_scope("MNIST_Dataset"):
    xs = tf.placeholder(tf.float32, [None, 784],name = "x_data")/255.   # 28x28
    ys = tf.placeholder(tf.float32, [None, 10],name = "y_data")
keep_prob = tf.placeholder(tf.float32,name="Dropout")
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

with tf.name_scope("ConvNN_Layer1"):
    W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
with tf.name_scope("Max_Pooling"):
    h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

with tf.name_scope("ConvNN_Layer2"):
    W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
with tf.name_scope("Max_Pooling"):
    h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64


with tf.name_scope("Full_Connection_Layer1"):
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope("Dropout"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


with tf.name_scope("Full_Connection_Layer2"):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="Predict_y_data")


# the error between prediction and real data
with tf.name_scope("Cross_Entropy"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar("Cross_Entorpy",cross_entropy)
with tf.name_scope("Train"):
    train = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
with tf.name_scope("Accuracy"):
    similiar_y_level = tf.equal(tf.arg_max(prediction,1),tf.arg_max(ys,1))
    accuracy = tf.reduce_mean(tf.cast(similiar_y_level,tf.float32))
tf.summary.scalar("Accuracy/MNIST_Data",accuracy)


""" 準備餵資料給框架 """
def feed_dict(feed_type = "train"):
    if feed_type == "train":
        batch_x,batch_y = mnist.train.next_batch(100)
        dropout = 0.5
        return {xs:batch_x,ys:batch_y,keep_prob:dropout}
    elif feed_type == "test":
        x_test_data,y_test_data = mnist.test.images[:1000],mnist.test.labels[:1000]
        dropout = 1.0
        return {xs:x_test_data,ys:y_test_data,keep_prob:dropout}

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("C:/board/train",sess.graph)
    test_writer = tf.summary.FileWriter("C:/board/test",sess.graph)
    tf.global_variables_initializer().run()
    for i in range(1001):
        if i % 50 == 0:
            summary,acc = sess.run([merged,accuracy],feed_dict=feed_dict("test"))
            test_writer.add_summary(summary,i)
            print("Acc",i//50,": ",acc)       
        else:
            summary,_ = sess.run([merged,train],feed_dict=feed_dict("train"))
            train_writer.add_summary(summary,i)

