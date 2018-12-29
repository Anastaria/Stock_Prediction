# Stock_Prediction
Stock price prediction based on LSTM, seq2seq and ARIMA.


This is the final project of Information Theory. In this task, we are required to predict minute wise stock price in the following one month based on the past 23 months. Here I assume the missing time period is caused by stock market suspension. Preprocessing is also needed since there is duplicated data.

Stock price is typical time series data. So, the first idea came to my mind is adopting LSTM (long short-term memory) to do the prediction. Based on the recurrent neural networks, LSTM model could solve the problem of long-range dependence thanks to the controlling gates, which make the model more flexible in remembering information.

Here I use the data of past 32 data points to predict the following 1 data point. In training data, the model has good performance when we only need to predict one time point. But once we need to use the predicted data as new input to do successive prediction, the prediction will end up with a line, which means the prediction shouldn’t be done step by step.

Then I borrow the idea from machine translation. We can build a sequence to sequence model, input the data of 32 points and output the data of the following 16 points. Attention mechanism is also included to assign different weights to hidden vector. But this model is too hard to convergence.

Finally, I move to traditional time series model, instead of deep learning ones. I choose autoregressive integrated moving average (ARIMA) model, which is actually made up of two methods, autoregressive and moving average. There are three parameters we need to decide in this mode: p,d,q, all of which are non-negative integers.
p is the order (number of time lags) of the autoregressive model.
d is the degree of differencing.(the number of times the data have had past values subtracted) .
q is the order of the moving-average model.

