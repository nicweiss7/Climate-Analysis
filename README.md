# Temperature Time Series Analysis

This project analyzes monthly global temperature anomalies from the HadCRUT5 dataset between 1850 and 2024.

First, LASSO regression is used to analyze the long-term temperature trend. Separate regressions for different time periods are also considered to compare changes in the trend over time.

![LASSO Regression](figures/lasso_regression.png)

The time series is then analyzed using ACF, PACF and the Augmented Dickey-Fuller test. Seasonal differencing with a lag of 12 months is applied to investigate stationarity.

![ACF](figures/ACF.png)
![PACF](figures/PACF.png)

Finally, an ARIMA model is fitted to the temperature data and evaluated for different forecast horizons.

![ARIMA Forecast](figures/forecast.png)

The forecasts should be interpreted with caution, as their plausibility is very limited by the simplicity of the model and the very long forecast horizon.

## Methods

- LASSO Regression
- ACF and PACF
- Augmented Dickey-Fuller Test
- Seasonal Differencing
- ARIMA Time Series Modelling

## Data

Monthly global temperature anomaly data from the HadCRUT5 dataset.
