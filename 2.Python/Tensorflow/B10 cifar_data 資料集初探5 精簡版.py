import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

train_file = "./data/CIFAR10_data/cifar-10-batches-py/data_batch_1" 
test_file = "./data/CIFAR10_data/cifar-10-batches-py/test_batch"
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def labels_one_hot(labels_num):
    label_one_hot = np.array(np.zeros([10],dtype=float))
    label_one_hot[labels_num]=1.0
    return label_one_hot     

train_data = unpickle(train_file)
test_data = unpickle(test_file)
print(len(test_data[b'filenames']))

print(labels_one_hot(train_data[b'labels'][0]))
plt.imshow(np.reshape(test_data[b'data'][0],(3, 32, 32)).transpose(1, 2, 0))
#for i in [0,3,999,9998,9999]:  
#    print("第",i,"張圖的標籤為:",train_data[b'labels'][i])
#    plt.imshow(np.reshape(train_data[b'data'][i],(3, 32, 32)).transpose(1, 2, 0))
#    plt.draw()
#    plt.pause(1)
#    plt.close()
    #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) # 將深度轉置 RGB --> BGR
    #cv2.namedWindow("wdw",0)  
    #cv2.imshow("img"+i,img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()



