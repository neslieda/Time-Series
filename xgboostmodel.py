import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'C:\Users\edayu\PycharmProjects\Yapayzeka\tripy\test (1).csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(data.head())

# Convert the date column to datetime format
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M:%S')

# Aggregate the data by date and region to get daily counts
aggregated_data = data.groupby(['date', 'name']).sum().reset_index()

# Display the first few rows of the aggregated data
print(aggregated_data.head())

# Create time-based features
aggregated_data['day_of_week'] = aggregated_data['date'].dt.dayofweek
aggregated_data['month'] = aggregated_data['date'].dt.month
aggregated_data['day_of_month'] = aggregated_data['date'].dt.day

# Create lag features
def create_lag_features(df, lag=1):
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df.groupby('name')['count'].shift(i)
    return df

# Apply lag features
lag_features_data = create_lag_features(aggregated_data.copy(), lag=7)

# Drop rows with NaN values caused by lagging
lag_features_data = lag_features_data.dropna().reset_index(drop=True)

# Display the first few rows of the dataframe with lag features
print(lag_features_data.head())

from sklearn.model_selection import train_test_split

# Özellikler ve hedef değişkeni tanımlayın
features = ['day_of_week', 'month', 'day_of_month'] + [f'lag_{i}' for i in range(1, 8)]
target = 'count'

# Verileri eğitim ve test setlerine ayırın
X = lag_features_data[features]
y = lag_features_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# XGBoost DMatrix formatına dönüştürme
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Model parametrelerini ayarlama
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'eta': 0.1,
}

# Modeli eğitme
model = xgb.train(params, dtrain, num_boost_round=100)

# Test setinde tahmin yapma
y_pred = model.predict(dtest)

# Modeli değerlendirme
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")


# Tahmin ve gerçek değerleri görselleştirme
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Gerçek Değerler')
plt.plot(y_pred, label='Tahmin Edilen Değerler')
plt.legend()
plt.xlabel('Zaman')
plt.ylabel('Bisiklet Sayısı')
plt.title('Gerçek ve Tahmin Edilen Bisiklet Sayıları')
plt.show()