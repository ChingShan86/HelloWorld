import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope("Layer"):
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
        return outputs

# 觀察可是化流程圖 可不需要輸入資料
#x_datas = np.linspace(-1,1,300)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, x_data.shape)
#y_datas = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope("Input_Data"): # <--建構一個大框框圖 裡面包含了 x_data , y_real_data
    x_data = tf.placeholder(tf.float32, [None, 1],name = 'x_data')
    y_real_data = tf.placeholder(tf.float32, [None, 1],name = 'y_real_data')

# 於自訂函式定義好塗層框架後，之後每家一個神經層 系統會連帶加上可視畫圖層
l1 = add_layer(x_data, 1, 10, activation_function=tf.nn.relu)
y_prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_real_data - y_prediction),reduction_indices=[1]),name = "loss")
with tf.name_scope("train"):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


with tf.Session() as sess:
    # 編譯完成後 到cmd等命令列 cd 至 board資料夾的上一層資料夾 運行 tensorboard --logdir board
    """ 
    成功方法 !!! 
    可行1:
        將 boardy 資料夾直接複製到 C:\ 下 
        然後再anaconda cmd命令列 運行  tensorboard --logdir board\ --port=6006 在網頁輸入 http:\\sam:6006
    可行2:
        可將port改成自訂port 8000 或等等 在網頁輸入 http:\\sam:8000
    能成功的命令寫法:
        ** 若檔案在 C:\ 下 可以不用寫 --port=6006 ， 於github資料夾下怎樣都會行不通
        沒寫等於
        1. tensorboard --logdir board ... <--- 最簡潔
        2. tensorboard --logdir board\ ...
        3. tensorboard --logdir "board" ...  
        4. tensorboard --logdir "board\" ... <---- 不會成功顯示 都是空白的
        有寫等於
        1. tensorboard --logdir=board ...
        2. tensorboard --logdir=board\ ...  <-- 比較直觀
        3. tensorboard --logdir="board" ...
        4. tensorboard --logdir="board\" ... <---- 不會成功顯示 都是空白的
    問題2: 每次重新編譯時 會累積重複之前的圖表
    解決2: 只要restart kernel in the spyder 即可
    """
    writer = tf.summary.FileWriter("C:/board/",sess.graph)
    tf.global_variables_initializer().run()
    # 觀察可是化流程圖 可不需要訓練資料
#    for i in range(1000):
#        # training
#        sess.run(train, feed_dict={x_data: x_data, y_real_data: y_data})
#        if i % 50 == 0:
#            # to see the step improvement
#            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
