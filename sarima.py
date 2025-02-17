import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv(r'C:\Users\edayu\PycharmProjects\Yapayzeka\tripy\test (1).csv')
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M:%S')

data = data.sort_values(by='date')


region_name = 'Eskişehir_1'
region_data = data[data['name'] == region_name]
region_data.set_index('date', inplace=True)

# Zaman serisi verisini görselleştir
plt.figure(figsize=(12, 6))
plt.plot(region_data['count'], label='Count')
plt.title(f'{region_name} Count Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()

p = d = q = range(0, 2)
pdq = list(product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

best_aic = np.inf
best_param = None
best_seasonal_param = None

for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            mod = SARIMAX(region_data['count'],
                          order=param,
                          seasonal_order=seasonal_param,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results = mod.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_param = param
                best_seasonal_param = seasonal_param
        except:
            continue

print(f'Best SARIMA model: order={best_param}, seasonal_order={best_seasonal_param}, AIC={best_aic}')

best_model = SARIMAX(region_data['count'],
                     order=best_param,
                     seasonal_order=best_seasonal_param,
                     enforce_stationarity=False,
                     enforce_invertibility=False)
best_model_fit = best_model.fit(disp=False)

forecast_steps = 30
forecast = best_model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=region_data.index[-1], periods=forecast_steps + 1, freq='D')[1:]

plt.figure(figsize=(12, 6))
plt.plot(region_data['count'], label='Actual')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', linestyle='--')
plt.title(f'SARIMA Model Forecast for {region_name}')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()
