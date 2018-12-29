# Stock_Prediction
Stock price prediction based on LSTM, seq2seq and ARIMA
In this task, we are required to predict minute wise stock price in the following one month based on the past 23 months. We assume the missing time period is caused by stock market suspension. Preprocessing is also needed since there is duplicated data.

Stock price is typical time series data. So, the first idea came to my mind is adopting LSTM (long short-term memory) to do the prediction. Based on the recurrent neural networks, LSTM model could solve the problem of long-range dependence thanks to the controlling gates, which make the model more flexible in remembering information.

Here we use the data of past 32 data points to predict the following 1 data point. In training data, the model has good performance when we only need to predict one time point. But once we need to use the predicted data as new input to do successive prediction, the prediction will end up with a line, which means the prediction shouldn’t be done step by step.

Then I borrow the idea from machine translation. We can build a sequence to sequence model, input the data of 32 points and output the data of the following 16 points. Attention mechanism is also included to assign different weights to hidden vector. But this model is too hard to convergence.

Finally, we move to traditional time series model, instead of deep learning ones. We choose autoregressive integrated moving average (ARIMA) model, which is actually made up of two methods, Autoregressive and Moving Average. There are three parameters we need to decide in this mode. P,d,q
