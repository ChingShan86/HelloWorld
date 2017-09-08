import tensorflow as tf
from PIL import Image
import numpy as np
import os

IMG_SIZE = 128 #图像大小
LABEL_CNT = 2 #标签类别的数量
P_KEEP_INPUT = 0.8 #输入dropout层保持比例
P_KEEP_HIDDEN = 0.5 #隐层dropout的保持比例

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #return filename and example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 # normalize
    label = tf.cast(features['label'], tf.int32)
    label = tf.sparse_to_dense(label, [2], 1, 0)

    return img, label

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_and_decode(filename_queue)
    min_after_dequeue = 1000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity, num_threads = num_threads,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

# 获取并初始化权重
def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

X = tf.placeholder("float", [None, IMG_SIZE, IMG_SIZE, 3])
Y = tf.placeholder("float", [None, 2])

w = init_weights([3, 3, 3, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([3, 3, 128, 128])
w5 = init_weights([4 * 4 * 128, 625])
w_o = init_weights([625, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# 简单的卷积模型
def simple_model(X, w, w_2, w_3, w_4, w_5, w_o, p_keep_input, p_keep_hidden):
    # batchsize * 128 * 128 * 3
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    # 2x2 max_pooling
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #dropout
    l1 = tf.nn.dropout(l1, p_keep_input) # 64 * 64 * 32
    
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w_2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_hidden) # 32 * 32 * 64
    
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_hidden) # 16 * 16 * 128
    
    l4a = tf.nn.relu(tf.nn.conv2d(l3, w_4, strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME') # 4 * 4 * 128
    l4 = tf.reshape(l4, [-1, w_5.get_shape().as_list()[0]])
    
    l5 = tf.nn.relu(tf.matmul(l4, w_5))
    l5 = tf.nn.dropout(l5, p_keep_hidden)

    return tf.matmul(l5, w_o)

#y_pred是预测tensor
y_pred = simple_model(X, w, w2, w3, w4, w5, w_o, p_keep_input, p_keep_hidden)

#定义损失函数为交叉熵。
#注意simple_model最后返回的不包含softmax操作，
# softmax_cross_entropy_with_logits会自动做softmax。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred))
#定义精度
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(y_pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#每batch随机取500张
test_img_batch, test_label_batch = input_pipeline(["./test.bin"], 100)
#tf.reset_default_graph()
with tf.Session() as sess:
    # 加载模型。模型的文件名称看下本地情况
    saver = tf.train.Saver()
    saver.restore(sess, './graph.ckpt-500')

    coord_test = tf.train.Coordinator()
    threads_test = tf.train.start_queue_runners(coord=coord_test)
    test_imgs, test_labels = sess.run([test_img_batch, test_label_batch])
    #预测阶段，keep取值均为1
    acc = sess.run(accuracy, feed_dict = {X : test_imgs, Y : test_labels, p_keep_hidden: 1.0, p_keep_input: 1.0})
    print("predict accuracy is %.3f" % acc)
    coord_test.request_stop()
    coord_test.join(threads_test)
