#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:48:34 2026

@author: nicoweiss
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.arima.model import ARIMA


#%% Data

data = pd.read_csv('/Users/nicoweiss/Desktop/Optimization for Data Analysis/Data/D1_HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv')

data["time"] = pd.to_datetime(data["Time"])
temperature = np.array(data["Anomaly (deg C)"])

data["x"] = (
    (data["time"] - pd.Timestamp("1850-01-01")).dt.days
    / 365.25
)

x = data["x"].to_numpy()
y = temperature


#%% LASSO regression

lam = 1

def objective(theta):

    a = theta[0]
    beta = theta[1]

    return np.sum((a * x + beta - y)**2) + lam * np.abs(a)


result = minimize(
    objective,
    x0=np.array([0.0, 0.0]),
    method="Powell"
)

a = result.x[0]
beta = result.x[1]

print("a =", a)
print("beta =", beta)

y_pred = a * x + beta

plt.figure(figsize=(10, 5))
plt.plot(data["time"], y, marker=".", label="Temperature")
plt.plot(data["time"], y_pred, linewidth=2, label="LASSO regression")
plt.title("Earth Temperature 1850 - 2024")
plt.xlabel("Date")
plt.ylabel("Temperature anomaly")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% Regression for different time periods

data["period"] = pd.cut(
    data["time"].dt.year,
    bins=[1850, 1900, 1940, 1975, np.inf],
    right=False,
    labels=["1850-1900", "1900-1940", "1940-1975", "1975-end"]
)

plt.figure(figsize=(10, 5))

plt.plot(
    data["time"],
    data["Anomaly (deg C)"],
    marker=".",
    label="Temperature"
)

plt.plot(
    data["time"],
    y_pred,
    linewidth=2,
    label="LASSO regression"
)

for period, subset in data.groupby("period", observed=True):

    print("")
    print(period, len(subset))

    x_sub = subset["x"].to_numpy()
    y_sub = subset["Anomaly (deg C)"].to_numpy()

    def objective_sub(theta):

        a_sub = theta[0]
        beta_sub = theta[1]

        return (
            np.sum((a_sub * x_sub + beta_sub - y_sub)**2)
            + lam * np.abs(a_sub)
        )

    result = minimize(
        objective_sub,
        x0=np.array([0.0, 0.0]),
        method="Powell"
    )

    a_sub = result.x[0]
    beta_sub = result.x[1]

    y_pred_sub = a_sub * x_sub + beta_sub

    plt.plot(
        subset["time"],
        y_pred_sub,
        linewidth=3,
        label=f"Regression {period}"
    )

    print("a =", a_sub)
    print("beta =", beta_sub)

plt.title("Earth Temperature 1850 - 2024")
plt.xlabel("Date")
plt.ylabel("Temperature anomaly")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% ACF and PACF

lags = 30
T = len(y)

ACF = acf(y, nlags=lags, fft=False)
PACF = pacf(y, nlags=lags)

conf_interval = 2 / np.sqrt(T)

plt.figure(figsize=(10, 5))
plt.stem(range(lags + 1), ACF)
plt.hlines(
    [conf_interval, -conf_interval],
    xmin=0,
    xmax=lags,
    linestyles="dashed"
)
plt.title("Autocorrelation Function (ACF)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.stem(range(lags + 1), PACF)
plt.hlines(
    [conf_interval, -conf_interval],
    xmin=0,
    xmax=lags,
    linestyles="dashed"
)
plt.title("Partial Autocorrelation Function (PACF)")
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Seasonal differencing

S = 12

y_new = y[S:] - y[:-S]
time_new = data["time"].iloc[S:]

plt.figure(figsize=(10, 5))
plt.plot(time_new, y_new, marker=".")
plt.title(f"Differences for Seasonality S={S}: Earth Temperature 1850-2024")
plt.xlabel("Date")
plt.ylabel("Temperature difference")
plt.grid(True)
plt.tight_layout()
plt.show()


#%% ADF test

adf_original = adfuller(y)

print("")
print("Original time series:")
print("ADF statistic:", adf_original[0])
print("p-value:", adf_original[1])

adf_diff = adfuller(y_new)

print("")
print("Seasonally differenced time series:")
print("ADF statistic:", adf_diff[0])
print("p-value:", adf_diff[1])


#%% ACF and PACF after differencing

ACF_new = acf(y_new, nlags=lags, fft=False)
PACF_new = pacf(y_new, nlags=lags)

conf_interval = 2 / np.sqrt(len(y_new))

plt.figure(figsize=(10, 5))
plt.stem(range(lags + 1), ACF_new)
plt.hlines(
    [conf_interval, -conf_interval],
    xmin=0,
    xmax=lags,
    linestyles="dashed"
)
plt.title("ACF of Seasonally Differenced Data")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.stem(range(lags + 1), PACF_new)
plt.hlines(
    [conf_interval, -conf_interval],
    xmin=0,
    xmax=lags,
    linestyles="dashed"
)
plt.title("PACF of Seasonally Differenced Data")
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.grid(True)
plt.tight_layout()
plt.show()


#%% ARIMA evaluation

test_years = [5, 10, 15, 20]

for years in test_years:

    test_size = years * 12
    stop = len(y) - test_size

    y_training = y[:stop]
    y_test = y[stop:]

    model = ARIMA(y_training, order=(2, 1, 2))
    fit = model.fit()

    y_pred = fit.forecast(steps=len(y_test))

    MSE = np.mean((y_test - y_pred)**2)

    print("")
    print(
        f"Mean Squared Error for a forecast horizon of "
        f"{years} years: {MSE}"
    )


#%% Final forecast

model_final = ARIMA(y, order=(2, 1, 2))
fit_final = model_final.fit()

print(fit_final.summary())

forecast_dates = pd.date_range(
    start=data["time"].iloc[-1] + pd.offsets.MonthBegin(1),
    end="2100-12-01",
    freq="MS"
)

steps = len(forecast_dates)

forecast = fit_final.forecast(steps=steps)

plt.figure(figsize=(12, 6))

plt.plot(
    data["time"],
    y,
    label="Observed Data"
)

plt.plot(
    forecast_dates,
    forecast,
    label="ARIMA Forecast"
)

plt.xlabel("Year")
plt.ylabel("Temperature anomaly")
plt.title("Forecast until 2100")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()