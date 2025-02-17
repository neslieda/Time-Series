import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Veriyi oku ve tarih sütununu datetime formatına çevir
data = pd.read_csv(r'C:\Users\edayu\PycharmProjects\Yapayzeka\tripy\test (1).csv ')
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M:%S')

# Verileri sırala
data = data.sort_values(by='date')

# Benzersiz bölgeleri al
regions = data['name'].unique()

# Her bölge için zaman serisi analizi yap
for region in regions:
    region_data = data[data['name'] == region]
    region_data.set_index('date', inplace=True)

    # Zaman serisi verisini görselleştir
    plt.figure(figsize=(12, 6))
    plt.plot(region_data['count'], label='Count')
    plt.title(f'{region} Count Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ARIMA modeli oluştur ve eğit
    model = ARIMA(region_data['count'], order=(1, 1, 1))
    model_fit = model.fit()

    # Tahminleri görselleştir
    forecast = model_fit.forecast(steps=10)
    plt.figure(figsize=(12, 6))
    plt.plot(region_data['count'], label='Actual')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title(f'ARIMA Model Forecast for {region}')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.show()


