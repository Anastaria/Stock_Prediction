import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import copy
import csv
import pickle

# 定义常量
rnn_unit = 10
input_size = 3  # 每一条数据的维度
output_size = 1
step = 32
batch_size = 128
lr = 1e-5
# ——————————————————导入数据——————————————————————
f = open('/Users/Anna/Desktop/traing_data_new.csv')
df = pd.read_csv(f)  # 读入股票数据 shape=(55500854, 9)
df.pop('Last Closing Price')
df.pop('Date')
df.pop('Time')
df = df.fillna(method="ffill")
#data = df.iloc[:20000, 4:8].values  # 取出有用的数据列
#data = df.iloc[:, 1:6].values  # 取出有用的数据列
data = df.iloc[:, 1:6]
train_portion=0.8

def get_data_all(batch_size,time_step):
    temp = data.iloc[0, 0]
    print("the current stock is:%s" % temp)
    begin=0
    end=0
    train_x_all=[]
    train_y_all=[]
    test_x_all = []
    test_y_all = []
    for i in data['Stock Code'].values:
        if i==temp:
            end+=1
        if (i!=temp) | (end==len(data)):
            single_stock=data.iloc[begin:end,1:].values
            train_x,train_y=get_train_data(single_stock,batch_size,time_step)
            train_x_all.extend(train_x)
            train_y_all.extend(train_y)
            test_x,test_y=get_test_data(single_stock,time_step)
            test_x_all.extend(test_x)
            test_y_all.extend(test_y)
            begin=end
            end+=1
            temp=i
            print("the current stock is:%s"%i)
    train_x_all_array=np.array(train_x_all).reshape(-1,input_size)
    normalized_train_x = (train_x_all_array - np.mean(train_x_all_array, axis=0)) / np.std(train_x_all_array, axis=0)
    normalized_train_x=normalized_train_x.reshape(-1,batch_size,time_step,input_size)
    train_y_all_array = np.array(train_y_all).reshape(-1, output_size)
    mean_y=np.mean(train_y_all_array, axis=0)
    std_y=np.std(train_y_all_array, axis=0)
    normalized_train_y = (train_y_all_array - mean_y) / std_y
    normalized_train_y = normalized_train_y.reshape(-1, batch_size,output_size)


    test_x_all_array=np.array(test_x_all).reshape(-1,input_size)
    normalized_test_x = (np.array(test_x_all) - np.mean(test_x_all_array, axis=0)) / np.std(test_x_all_array, axis=0)  # 标准化
    normalized_test_x = normalized_test_x.reshape(-1, time_step, input_size)
    test_y_all_array = np.array(test_y_all)
    normalized_test_y = (np.array(test_x_all) - np.mean(test_y_all_array, axis=0)) / np.std(test_y_all_array, axis=0)

    return normalized_train_x,normalized_train_y,normalized_test_x,normalized_test_y,mean_y,std_y


# 获取训练集
def get_train_data(single_stock,batch_size,time_step):
    train_begin=0
    train_end=int(single_stock.shape[0] * train_portion)
    data_train = single_stock[train_begin:train_end,:]
    train_x, train_y = [], []
    batch_x, batch_y = [], []
    temp_x, temp_y = [], []
    length=len(data_train) - time_step + 1
    for i in range(length):
        x = data_train[i:i + time_step, :input_size]
        y = data_train[i + time_step-1:i + time_step, input_size, np.newaxis]  # 标签：该time step中最后一个时间点的收盘价
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
def get_test_data(single_stock,time_step):
    test_begin= int(single_stock.shape[0] * train_portion)
    data_test = single_stock[test_begin:single_stock.shape[0],:]
    normalized_test_data=data_test
    test_x, test_y = [], []
    for i in range(len(normalized_test_data) - time_step + 1):
        x = normalized_test_data[i:i + time_step, :input_size]
        y = normalized_test_data[i + time_step-1:i + time_step, input_size]  # 标签：该time step中最后一个时间点的收盘价
        test_x.append(x.tolist())
        test_y.append(y.tolist())
    return test_x, test_y

train_x_all,train_y_all,normalized_test_data,test_y_all,mean_y,std_y=get_data_all(batch_size,step)
tempdata=[train_x_all,train_y_all,normalized_test_data,test_y_all,mean_y,std_y]
fw=open('dataFile1.pkl','wb')
pickle.dump(tempdata,fw)
fw.close()