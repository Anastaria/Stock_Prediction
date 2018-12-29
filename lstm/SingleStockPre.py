import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import copy
import csv

# 定义常量
rnn_unit = 10
input_size = 3  # 每一条数据的维度
output_size = 1
time_step = 32
batch_size = 128
lr = 1e-5
# ——————————————————导入数据——————————————————————
f = open('traing_data_new.csv')
df = pd.read_csv(f)  # 读入股票数据 shape=(55500854, 9)
df.pop('Last Closing Price')
df= df.fillna(method="ffill")
data = df.iloc[:10000, 4:8].values  # 取出有用的数据列
num_data = data.shape[0]
train_end = int(num_data * 0.8)


# 获取训练集
def get_train_data(batch_size,time_step, train_begin, train_end):
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    train_x, train_y = [], []
    batch_x, batch_y = [], []
    temp_x, temp_y = [], []
    length=len(normalized_train_data) - time_step + 1
    for i in range(length):
        x = normalized_train_data[i:i + time_step, :3]
        y = normalized_train_data[i + time_step-1:i + time_step, 3, np.newaxis]  # 标签：该time step中最后一个时间点的收盘价
        batch_x.append(x.tolist())
        batch_y.append(y.tolist())
        if (i==(batch_size-(length%batch_size)-1)):
            temp_x=copy.copy(batch_x)
            temp_y=copy.copy(batch_y)
        if (i % batch_size == batch_size-1) & (i > 0):
            train_x.append(batch_x)
            train_y.append(batch_y)
            batch_x, batch_y = [], []
        if (i==(length-1)) & (i%batch_size!=batch_size -1):
            batch_x.extend(temp_x)
            batch_y.extend(temp_y)
            train_x.append(batch_x)
            train_y.append(batch_y)
    return train_x, train_y


# 获取测试集 前time step-1个没有预测结果
def get_test_data(time_step=time_step, test_begin=train_end + 1):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    test_x, test_y = [], []
    for i in range(len(normalized_test_data) - time_step + 1):
        x = normalized_test_data[i:i + time_step, :3]
        y = normalized_test_data[i + time_step-1:i + time_step, 3]  # 标签：该time step中最后一个时间点的收盘价
        test_x.append(x.tolist())
        test_y.extend(y)
    return mean, std, test_x, test_y


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X,mybatch_size, mytime_step):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in  #according a full connect layer to convert input_size->rnn_uni
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)#调用构造函数，构造对象cell
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    indices = [[i, mytime_step-1] for i in range(mybatch_size)]  # the index with last cell
    output = tf.gather_nd(output_rnn, indices)
    output=tf.reshape(output,[-1,rnn_unit]) #作为输出层的输入

    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states #每个time step只保留最后一个神经元的输出

# ——————————————————训练模型——————————————————
def train_lstm(batch_size=batch_size, time_step=time_step, train_begin=0, train_end=train_end):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, 1, output_size])
    train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):  # 由于重复使用参数，需要设定命名空间
        pred, _ = lstm(X,batch_size, time_step)
    # 损失函数
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1]))))  # RMSE均方根误差
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)  # 用saver对象保存模型的参数
    # module_file = tf.train.latest_checkpoint('stock2.model')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        # 重复训练10000次
        for i in range(10000):
            index=random.randint(0,(len(train_x)-1))
            _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[index],Y:train_y[index]})


            if i % 500 == 0:
                print("保存模型：", saver.save(sess, 'model_save2/model.ckpt', global_step=i))
                print(i, loss_)

#train_lstm()


# ————————————————预测模型————————————————————
def prediction(time_step=time_step):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    # Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(X,1,time_step)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[3] + mean[3]  # 恢复归一化处理
        test_predict = np.array(test_predict) * std[3] + mean[3]
        rmse = np.sqrt(np.average(np.square(test_predict - test_y)))
        print("The rmse of this predict:", rmse)
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


prediction()
