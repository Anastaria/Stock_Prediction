#import tushare as ts #财经数据接口包
import pandas as pd
import numpy as np
import datetime
import copy
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle

sns.set_style("whitegrid", {"font.sans-serif": ['KaiTi', 'Arial']})#设置主题

'''
fr=open('StockFile.pkl','rb')
data1=pickle.load(fr)
df,num_stock,stock_code=data1[0],data1[1],data1[2]
'''
# the argument
FREQ = '3D'


# fr=open('StockFile.pkl','rb')
# data1=pickle.load(fr)
# df,stock_code=data1[0],data1[1]

f = open('traing_data_new.csv')
df = pd.read_csv(f)  # 读入股票数据 shape=(55500854, 9)
df1=df.iloc[:100000,:]
df1['datetime']=df1['Date']+'-'+df1['Time']
df1.index=df1['datetime']
df1.drop(df1.columns[[0,1,2,3,5,9]], axis=1, inplace=True)
df1.columns=['open','high','low','close']
def run_main():
    #k = ts.get_hist_data('002867', start='2017-04-10', end='2018-06-20')
    # 002867茅台股票  这里可以设置获取的时间段

    #lit = ['open', 'high', 'close', 'low']  # 这里我们只获取其中四列
    #data = df1[lit]
    data=df1
    d_one = data.index  # 以下9行将object的index转换为datetime类型
    d_two = []
    d_three = []
    date2 = []
    for i in d_one:
        d_two.append(i)
    for i in range(len(d_two)):
        d_three.append(parse(d_two[i]))
    data2 = pd.DataFrame(data, index=d_three,
                         dtype=np.float64)  # 构建新的DataFrame赋予index为转换的d_three。当然你也可以使用date_range()来生成时间index

    data2 = data2.drop_duplicates(keep='first')
    data2 = data2.sort_index(axis=0)
    plt.plot(data2['close'])
    # 显然数据非平稳，所以我们需要做差分
    plt.title('股市每日收盘价')
    #plt.show()

    data2_w = data2['close'].resample(FREQ).mean()
    # 由于原始数据太多，按照每一周来采样，更好预测，并取每一周的均值
    data2_train = data2_w['2008-01':'2009-11']  # 我们只取2017到2018的数据来训练
    plt.plot(data2_train)
    plt.title('周重采样数据')
    #plt.show()
    data2_train = data2_train.dropna(axis=0, how='any')
    new_index = pd.date_range('20180101', periods=len(data2_train),freq = FREQ)
    data2_train = pd.DataFrame(data2_train)
    data_train = copy.copy(data2_train)
    data2_train.set_index(new_index, inplace=True)
    # 一阶差分，分析ACF
    acf = plot_acf(data2_train, lags=20)  # 通过plot_acf来查看训练数据，以便我们判断q的取值
    plt.title("股票指数的 ACF")
    # acf.show()

    # 一阶差分，分析PACF
    pacf = plot_pacf(data2_train, lags=20)  # 通过plot_pacf来查看训练数据，以便我们判断p的取值
    plt.title("股票指数的 PACF")
    # pacf.show()
    data2_diff = data2_train.diff(1)  # 差分很简单使用pandas的diff()函数可以进行一阶差分
    diff = data2_diff.dropna()
    for i in range(1):  # 五阶差分，一般一到二阶就行了，我有点过分
        diff = diff.diff(1)
        diff = diff.dropna()
    plt.figure()
    plt.plot(diff)
    plt.title('2阶差分')
    #plt.show()

    # 五阶差分的ACF
    acf_diff = plot_acf(diff, lags=40)
    plt.title("2阶差分的ACF")  # 根据ACF图，观察来判断q
    # acf_diff.show()

    # 五阶差分的PACF
    pacf_diff = plot_pacf(diff, lags=40)  # 根据PACF图，观察来判断p
    plt.title("2阶差分的PACF")
    pacf_diff.show()

    # # 根据ACF和PACF以及差分 定阶并建模
    # model = ARIMA(data2_train, order=(3, 2, 4), freq='1D')  # pdq    频率按周
    #
    # # 拟合模型
    # arima_result = model.fit()
    #
    # # 预测
    # pred_vals = arima_result.predict('2018-08-20','2019-01-30', dynamic=True, typ = "levels")  # 输入预测参数，这里我们预测2017-01-02以后的数据
    # forcast_vals = arima_result.forecast(30)[0]
    # fore_new_index = pd.date_range('20190101', periods=len(forcast_vals))
    # forcast_vals = pd.DataFrame(forcast_vals)
    # forcast_vals.set_index(fore_new_index, inplace=True)
    # # 可视化预测
    # stock_forcast = pd.concat([data2_train, pred_vals], axis=1, keys=['original', 'predicted'])  # 将原始数据和预测数据相结合，使用keys来分层
    # fore_stock_forcast = pd.concat([data2_train, forcast_vals], axis=1, keys=['original', 'predicted'])  # 将原始数据和预测数据相结合，使用keys来分层
    data2_train_fit = data2_train[0:(len(data2_train) - 30)]
    # 根据ACF和PACF以及差分 定阶并建模
    model = ARIMA(data2_train_fit, order=(5, 2, 2), freq=FREQ)

    pred_begin = pd.date_range('20180101', periods=len(data2_train) - 30, freq = FREQ)[-1]    # 拟合模型
    arima_result = model.fit()

    # 预测
    # pred_vals = arima_result.predict('2018-08-20', '2019-01-30', dynamic=True,
    #                                  typ="levels")  # 输入预测参数，这里我们预测2017-01-02以后的数据
    forcast_vals = arima_result.forecast(30)[0]
    fore_new_index = pd.date_range(pred_begin, periods=len(forcast_vals))
    forcast_vals = pd.DataFrame(forcast_vals)
    forcast_vals.set_index(fore_new_index, inplace=True)
    # 可视化预测
    # stock_forcast = pd.concat([data2_train, pred_vals], axis=1,
    #                           keys=['original', 'predicted'])  # 将原始数据和预测数据相结合，使用keys来分层
    fore_stock_forcast = pd.concat([data2_train, forcast_vals], axis=1,
                                   keys=['original', 'predicted'])  # 将原始数据和预测数据相结合，使用keys来分层
    # 构图
    # plt.figure()
    # plt.plot(stock_forcast)
    # plt.title('真实值vs预测值')
    # plt.show()
    plt.figure()
    plt.plot(fore_stock_forcast)
    plt.title("Forcast Results")
    plt.show()
    a  = 1


if __name__ == "__main__":
    run_main()