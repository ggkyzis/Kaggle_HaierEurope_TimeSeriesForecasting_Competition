# Time-Series-Forecasting
My solution for Haier Europe Forecasting Hackathon with Python.  The goal is to forecast short-term 12 weeks weekly product level forecast and mid-term 3 months monthly aggregated level of the forecast.

My implementation initially tried to tackle an AutoML and Low-Code approach based on statistical forecasting and low-complexity algorithms. 
Afterwards, I transitioned to XGBRegressor and LGBMRegressor that indicate higher complexity but are state-of-the-art forecasting algorithms. My experiments had different number of lags and parameters as I performed a Bayesian Search in order to find optimal hyperparameters.

Due to the low volume of data in each time-series and the large volume of different time series to examine, Deep Learning approaches with LSTM, Transformer etc. were not examined.

My preference language was Python, and due to the high complexity of the problem I performed error-handling in each model fitting process in order to automate the forecasting of each univariate time series, and enable the completeness of my solution in time.
