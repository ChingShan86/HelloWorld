import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


filedir = "./data/CIFAR10_data/cifar-10-batches-py"
filename = "/data_batch_1" 
file = filedir + filename
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


img = unpickle(file)
print(type(img))
#print(img.__class__)
#print(img.__doc__)

#type(img['data']), type(img['labels']), type(img['batch_label']), type(img['filenames'])
""" 
print(img) 
{ b'labels':[6,8,9,....],
  b'data':array([[59,43,50....],[154,126,...],[...],[...],...,[]],dtype = unit8) 0~2^8-1整數
  b 'filenames':[b'apple_s_0004.png',b'...',b'...',.......] #1D array
  b 'batch_label': b'training batch 1 of 5'}
"""
print(img[b'batch_label'])
print(img[b'labels'])

print("---")

print(len(img[b'labels']))
print(len(img[b'data']))
print(len(img[b'data'][0]),",",len(img[b'data'][9999]))
print(len(img[b'filenames']))
print(len(img[b'batch_label']))

print("---")

print(img[b'labels'][0],",",img[b'labels'][9999])
print(img[b'data'][0],",",img[b'data'][9999])
print(img[b'filenames'][0],",",img[b'filenames'][9999])


im = tf.reshape(img[b'data'][0],shape=[-1,32,32,3])
#with tf.Session() as sess:
#    tf.global_variables_initializer().run()
#    print(sess.run(im))
#    print("---")
#    print(sess.run(im[0][0][0][0]))

"""
shape : (3,32,32)
transpose : (1,2,0)
代表 shape 索引 0 的值 移至 索引2的位置
    shape 索引 1 移至 索引0
    shape 索引 2 移至 索引1
結論:transpose中的位置擺放為該位置的值的新位置
"""
im3 = img[b'data']
print("dfhwh",im3.shape)
im2 = np.reshape(img[b'data'][0],(3, 32, 32)) #擺放位置為 深度RGB 圖片COLS(X軸向右) 圖片ROWS(Y軸向下)
print(im2.shape)
im2 = im2.transpose(1, 2, 0)  # 轉置為 圖片COLS(X軸向右) 圖片ROWS(Y軸向下) 深度RGB
plt.imshow(im2)
plt.show()

print("標籤為:",img[b'labels'][0])

im2 = cv2.cvtColor(im2,cv2.COLOR_RGB2BGR) # 將深度轉置 RGB --> BGR
cv2.namedWindow("wdw",0)  
cv2.imshow("wdw",im2)
cv2.waitKey(0)

cv2.destroyAllWindows()
plt.close()







