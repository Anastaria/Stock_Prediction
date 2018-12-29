#要预测共计23天的股票价格 23*240=5520
#每只股票的编码需不需要作为一个输入？暂时不需要，并不蕴含太多股票本身的特征，只有时序特征
#对512只股票打包进行预测
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import copy
import pickle
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq
import seq2seq.seq2seq_model
# 定义常量
rnn_unit = 10
input_size = 4  # 每一条数据的维度
output_size = 1
step_enc = 5520
step_dec = 5520
batch_size = 16
lr = 1e-5

# ——————————————————导入数据——————————————————————
fr=open('dataFile.pkl','rb')
data1=pickle.load(fr)
train_x_all,train_y_all,test_x_all,test_y_all,mean,std=data1[0],data1[1],data1[2],data1[3],data1[4],data1[5]

# ——————————————————训练模型——————————————————
def train_lstm(batch_size=batch_size, time_step_enc=step_enc,time_step_dec=step_dec):
    X = tf.placeholder(tf.float32, shape=[None, time_step_enc, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step_dec, output_size])
    with tf.variable_scope("seq2seq"):  # 由于重复使用参数，需要设定命名空间
        pred, _ = seq2seq(X,batch_size, time_step_enc,time_step_dec)
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
            index=random.randint(0,(len(train_x_all)-1))
            _,loss_=sess.run([train_op,loss],feed_dict={X:train_x_all[index],Y:train_y_all[index]})
            if i%50==0:
                print(i, loss_)
            if i % 500 == 0:
                print("保存模型：", saver.save(sess, 'model_seq2seq_save/model.ckpt', global_step=i))

train_lstm()

# ————————————————预测模型————————————————————
def prediction(time_step_enc=step_enc,time_step_dec=step_dec):
    X = tf.placeholder(tf.float32, shape=[None,time_step_enc, input_size])
    # Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    with tf.variable_scope("seq2seq", reuse=tf.AUTO_REUSE):
        pred, _ = seq2seq(X,batch_size,time_step_enc,time_step_dec)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_seq2seq_save')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x_all)):
            prob = sess.run(pred, feed_dict={X: [test_x_all[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y_all)
        test_predict = np.array(test_predict) * std + mean # 恢复归一化处理
        rmse = np.sqrt(np.average(np.square(test_predict - test_y)))
        print("The rmse of this predict:", rmse)
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


prediction()




