from meteostat import Stations, Daily
from datetime import datetime
import pandas as pd

latitude = 39.7767
longitude = 30.5206

stations = Stations()
station = stations.nearby(latitude, longitude).fetch(1)

station_id = station.index[0]

start = datetime(2021, 1, 1)
end = datetime(2025, 12, 31)


data = Daily(station_id, start, end)
data = data.fetch()


data.to_csv('eskisehir_2021_2025_weather_data.csv')

print(data.isna().sum())
print(data.columns)

prcp_median = data['prcp'].median()
data['prcp'].fillna(prcp_median, inplace=True)

data.to_csv('eskisehir_2021_2025_weather_data_filled.csv')

print(data.isna().sum())

data = data[['prcp', 'tavg', 'pres', 'wspd', 'wdir']]

data.reset_index(inplace=True)
data['date'] = pd.to_datetime(data['time'])
data.drop(columns=['time'], inplace=True)


test_data = pd.read_csv('test (1).csv')

test_data['date'] = pd.to_datetime(test_data['date'], dayfirst=True)
test_data = test_data.sort_values(by='date')

merged_data = pd.merge(test_data, data, on='date', how='left')

merged_data.to_csv('merged_test_data.csv', index=False)
print(merged_data.head())
print(merged_data.isna().sum())
print(merged_data.columns)
-print(merged_data.info())
