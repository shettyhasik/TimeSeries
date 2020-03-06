#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

#importing the dataset
data = pd.read_csv("ML-EMHYY.csv",parse_dates = ['DATE'])

plt.plot_date(x=data['DATE'] ,y=data['BAMLEMHBHYCRPIEY'])
plt.boxplot(data['BAMLEMHBHYCRPIEY'])

data.info()
data.isnull().sum()

data.sort_values(['DATE'], ascending = True, inplace = True)

data['DATE'] = pd.to_datetime(data['DATE'])

data.set_index(['DATE'], inplace = True)

y = data['BAMLEMHBHYCRPIEY'].resample('MS').mean()

y.describe()

#ADF - Chk for stationarity, Ha = Stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(y)
print('ADF Statistic: {}'.format(result[0]))
print('P-value: {}'.format(result[1]))
for key,values in result[4].items():
    print("{}: {:.3f}".format(key, values))

""" p is less then 0.05, hence rejecting null hypothesis
    data is stationary
"""
#p-value is 0.01 < 0.05, thus we can reject null hypothesis.
    
#==== Decomposing ====
# decomposing the origanl time series into trend, seasonality & residual
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(y)

plt.plot(y, label="Original")
plt.legend(loc='best')

#Trend
trend = decomposition.trend
plt.show()
plt.plot(trend, label="Trend")
plt.legend(loc='best')

#Seasonal
season = decomposition.seasonal
plt.show()
plt.plot(season, label="Seasonal")
plt.legend(loc="best")

#Residuals
residuals = decomposition.resid
plt.show()
plt.plot(residuals, label="Residuals")
plt.legend(loc="best")
    
# There is tred, seasonality and residual. Hence we go  for sarima model

#Forecast using ARIMA(p, d, q)

#plotting acf & pacf
sm.graphics.tsa.plot_acf(y.values.squeeze())
plt.show()

sm.graphics.tsa.plot_pacf(y.values.squeeze())
plt.show()

#Fitting the ARIMA model 
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(5,0,8),
                                seasonal_order=(5,0,8,1),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(10,10))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2018-01-01')
                              )

pred_ci = pred.conf_int()

ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label="One-step forward Forecast", 
                         alpha=.7, figsize=(12, 12))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.2)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
plt.title("One-year forward Forecast")
plt.legend()
plt.show()

#RMSE
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print("The RMSE of Forecast: {}".format(round(np.sqrt(mse), 2)))

'''
Our model was able to forecast average daily sales in the test set within
0.27 of the real sales. '''   

#Applying garch model
from arch import arch_model
#from arch import ConstantMean, GARCH, Normal
"""
model = ConstantMean(y)
am.volatility = GARCH(1, 0, 1)
am.distribution = Normal()

"""
# --- ARCH model ---
model = arch_model(y, vol='GARCH')
model_fit = model.fit()
print(model_fit.summary)

yhat = model_fit.forecast(start=pd.to_datetime('2000-01-01'))
plt.plot(yhat.variance)

