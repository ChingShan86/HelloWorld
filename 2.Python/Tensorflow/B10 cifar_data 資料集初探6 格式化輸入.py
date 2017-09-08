import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

file1 = "./data/CIFAR10_data/cifar-10-batches-py/data_batch_1"
file2  = "./data/CIFAR10_data/cifar-10-batches-py/data_batch_1"
file3  = "./data/CIFAR10_data/cifar-10-batches-py/data_batch_1"
file4  = "./data/CIFAR10_data/cifar-10-batches-py/data_batch_1"
file5  = "./data/CIFAR10_data/cifar-10-batches-py/data_batch_1"
test_file = "./data/CIFAR10_data/cifar-10-batches-py/test_batch"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def labels_one_hot(labels):
    label_one_hot = tf.one_hot(labels,depth=10,on_value=1.0,off_value=0.0,axis = -1)
    return label_one_hot

def combine_batches_data(data_type = "train"):
    if data_type == "train":
        b1,b2,b3,b4,b5 = unpickle(file1),unpickle(file2),unpickle(file3),unpickle(file4),unpickle(file5)
        img = np.concatenate((b1[b'data'],b2[b'data'],b3[b'data'],b4[b'data'],b5[b'data']))
        label = np.concatenate((b1[b'labels'],b2[b'labels'],b3[b'labels'],b4[b'labels'],b5[b'labels']))
        labelname = np.concatenate((b1[b'filenames'],b2[b'filenames'],b3[b'filenames'],b4[b'filenames'],b5[b'filenames']))
    elif data_type == "test":
        test = unpickle(test_file)
        img = test[b'data']
        label = test[b'labels']
        labelname = test[b'filenames']
    return img,label,labelname

def test_batch(test_img,test_label, batch_size):
    min_after_dequeue = 1000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [test_img, test_label], batch_size=batch_size, capacity=capacity, num_threads = num_threads,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

        

train_img,train_label,train_labelname = combine_batches_data()
test_img,test_label,test_labelname = combine_batches_data("test")


print(train_img)
print(train_label)
print(test_img)
print(test_label)
print(len(train_img))
#print(len(train_label))
print(len(test_img))
#print(len(test_label))

test_img_batch,test_label_batch = test_batch(test_img,test_label,100)
print(test_img_batch)
print(test_label_batch)

#with tf.Session() as sess:
#    tf.global_variables_initializer().run()
#    #label_on_hot = labels_one_hot(train_label)
#    print((sess.run(test_label_batch)))
    #print(len(sess.run))


#print(labels_one_hot(train_data[b'labels'][0]))
#plt.imshow(np.reshape(test_data[b'data'][0],(3, 32, 32)).transpose(1, 2, 0))
d = np.array(test_label)
print(d)

def one_hot_encode(np_array):
    return (np.arange(10) == np_array[:,None]).astype(np.float32)

c = one_hot_encode(d)
print(c)


def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
a = np.array([[[1,2,3],
               [4,5,6],
               [7,8,9]],
              [[10,11,12],
               [13,14,15],
               [16,17,18]],
               [[19,20,21],
                [22,23,24],
                [25,26,27]]])
b = np.array([3,5,7])
b_ = one_hot_encode(b)

xs,ys = next_batch(2,a,b_)
print(xs)
print(ys)

