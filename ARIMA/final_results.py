import pandas as pd
import numpy as np

f=open('testing_data.csv')
test=pd.read_csv(f)
g=open('FINAL_RESULTS.csv')
results=pd.read_csv(g)
test['Closing Price']=1

stock_list=test['Stock Code'].unique()
num_list=test['Stock Code'].value_counts(sort=False)
begin=0
end=0
results_list = np.array([])
for stock in stock_list:
    length=num_list.ix[stock]
    begin=end
    end=begin+length
    res_column = results.loc[:,str(stock)]
    #test['Closing Price'].iloc[begin:end]=results.ix[0:length-1,str(stock)]
    results_list=np.append(results_list,res_column.loc[0:length-1].values)
    #test.loc[begin:end-1,'Closing Price'] = res_column.loc[0:length-1]
test['Closing Price'] = results_list
test.to_csv('/Users/Anna/Desktop/result_0.csv')
