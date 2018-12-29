import pandas as pd
import numpy as np
import pickle

def main():
   fr1 = open("out/left_1.pkl", 'rb')
   left_code = pickle.load(fr1)
   df383 = pd.read_csv('out/RESULTS__383.csv')
   df121 = pd.read_csv('out/RESULTS129_left_121.csv')
   df7 = pd.read_csv('out/RESULTS8_left_7.csv')
   df_new = pd.concat([df383,df121,df7], axis = 1)
   df_new["600225"] = None
   fr = open('StockFile.pkl', 'rb')
   data1 = pickle.load(fr)
   df, stock_code = data1[0], data1[1]
   df_new = df_new[stock_code]
   df_new.to_csv("FINAL_RESULTS")
   return

if __name__ == "__main__":
    main()