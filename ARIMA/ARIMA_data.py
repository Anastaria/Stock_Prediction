import pandas as pd
import pickle

f = open('traing_data_new.csv')
df = pd.read_csv(f)  # 读入股票数据 shape=(55500854, 9)
#df.drop(df.columns[[0]], axis=1, inplace=True)

def get_single_stock(data):
    temp = data.iloc[0, 1]
    print("the current stock is:%s" % temp)
    begin=0
    end=0
    stock_data=[]
    stock_code=[]
    for i in data['Stock Code'].values:
        if i==temp:
            end+=1
        if (i != temp) | (end == len(data)):
            single_stock=data.iloc[begin:end,:].values
            stock_data.append(single_stock)
            stock_code.append(str(temp))
            begin=end
            end+=1
            temp=i
            print("the current stock is:%s"%i)

    return stock_data,stock_code

stock_data,stock=get_single_stock(df)
tempdata=[stock_data,stock]
fw=open('StockFile.pkl','wb')
pickle.dump(tempdata,fw)
fw.close()