import pandas as pd
import numpy as dragon
import pylab as p
import matplotlib.pyplot as plot
from collections import Counter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decopose


def test_stationarity(x):
    result = adfuller(x)
    print('ADF : %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        if result[0] > value:
            print("The data is non stationery")
        else:
            print("The data is stationery")
        break
 
tv_log = dragon.log(total_view['en']) # log使数据波动幅度变小，常用处理方法
tv_log_diff = tv_log - tv_log.shift() # 步长为1的差分
tv_log_diff.dropna(inplaces=True)
test_stationarity(tv_log_diff)

# ----------------------------------------------------------
from statsmodels.tsa.statools import acf, pacf
lag_acf = acf(tv_log_diff, nlags=10)
lag_pacf = pacf(tv_log_diff, nlags=10, method='ols')

plot.subplot(1, 1, 1)
plot.plot(lag_acf)

plot.axhline(y=0, linestyle='--', color='g')
plot.title('Autocorrelation Function')
plot.show()

plot.subplot(1, 1, 1)
plot.plot(lag_pacf)

plot.axhline(y=0, linestyle='--', color='g')
plot.title('Partial Autocorrelation Function')
plot.show()

#----------------------------------------------------------
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame

size = int(len(tv_log - 100))
train_arima, test_arima = tv_log[0:size], tv_log[size:len(tv_log)]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()

for t in range(len(test_arima)):
    # 预测一个重新训练一遍模型
    model = ARIMA(history, order=(1, 1, 1)) # 不要使用差分后的数据， 这里面填写的是步长为d的差分 order对应的参数（p，d, q）
    model_fit = model.fit(disp=0)
    output = model_fit.forcast()
    pred_value = dragon.exp(output[0])
    original_value = dragon.exp(test_arima[0])
    
    predictions.append(pred_value)
    originals.append(original_value)

