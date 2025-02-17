import os
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

df = pd.read_csv('merged_test_data.csv', index_col=0)
layer_size = 2
units = 30
SEED = 123
learning_rate = 0.001
layer = 'LSTM'
activation_function = 'relu'
dropout_rate = 0.1
epoch = 50
batch_size = 128
loss_function = 'mae'
n_days = 1
test_size = 10
target = 'count'

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
print("SEED Ayarlandı: ", SEED)

cols_to_analyze = [
    "count",
    "prcp",
    "tavg",
    "pres",
    "wspd",
    "wdir"
]

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

unique_regions = df['name'].unique()

region_rmses = []

if not os.path.exists('plots'):
    os.makedirs('plots')

for region in unique_regions:
    region_df = df[df['name'] == region]

    if len(region_df) < (n_days + test_size):
        print(f"{region} bölgesi için yeterli veri yok.")
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_df = pd.DataFrame(scaler.fit_transform(region_df[cols_to_analyze]),
                             index=region_df.index,
                             columns=cols_to_analyze)

    reframed_df = pd.DataFrame(series_to_supervised(scaled_df.to_numpy(), n_days, 1))
    reframed_df = reframed_df.iloc[:, :-(len(cols_to_analyze) - 1)]
    reframed_df.index = region_df.index[n_days:]

    X = reframed_df.iloc[:, :-1]
    y = reframed_df.iloc[:, -1]
    print(X.shape, "x", y.shape, "y")

    X_train_df = X[:-test_size]
    X_test_df = X[-test_size:]
    y_train_df = y[:-test_size]
    y_test_df = y[-test_size:]

    X_train = X_train_df.to_numpy().reshape(
        (X_train_df.shape[0], n_days, scaled_df.shape[1]))
    X_test = X_test_df.to_numpy().reshape(
        (X_test_df.shape[0], n_days, scaled_df.shape[1]))

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(
        LSTM(units=50,
             activation=activation_function))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation=activation_function))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)

    history = model.fit(X_train,
                        y_train_df.to_numpy(),
                        epochs=100,
                        batch_size=64,
                        validation_data=(X_test, y_test_df.to_numpy()),
                        verbose=1,
                        shuffle=False)

    scaler_pred = MinMaxScaler()
    scaler_pred.fit(region_df[target].to_numpy().reshape(-1, 1))
    y_pred_test = model.predict(X_test)
    inv_y_pred_test = scaler_pred.inverse_transform(y_pred_test)
    y_true_test = scaler_pred.inverse_transform(y_test_df.to_numpy().reshape(-1, 1))
    y_pred_train = model.predict(X_train)
    inv_y_pred_train = scaler_pred.inverse_transform(y_pred_train)
    y_true_train = scaler_pred.inverse_transform(y_train_df.to_numpy().reshape(-1, 1))

    test_rmse = mean_squared_error(inv_y_pred_test, y_true_test, squared=False)
    test_mae = mean_absolute_error(inv_y_pred_test, y_true_test)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_df.index, y_true_test, label='Gerçek Değerler')
    plt.plot(y_test_df.index, inv_y_pred_test, label='Tahmin Değerleri')
    plt.title(f'{region} Bölgesi için Tahmin Sonuçları')
    plt.xlabel('Tarih')
    plt.ylabel('Bicycle Count')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(f'plots/{region}_prediction.png')


    region_rmses.append(test_rmse)

    print(f'{region} Bölgesi - Test RMSE: {test_rmse}, Test MAE: {test_mae}')

general_test_rmse = np.mean(region_rmses)
print(f'Genel Test RMSE: {general_test_rmse}')

import seaborn as sns

for col in region_df.columns:
    if region_df[col].dtype == 'object':
        try:
            region_df[col] = pd.to_numeric(region_df[col], errors='coerce')
        except:
            pass

corr = region_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.savefig('corr_map.png')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
file_path = 'merged_test_data.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])
unique_names = data['name'].unique()
features = ["count", "prcp", "tavg", "pres", "wspd", "wdir"]
output_dir = 'ACF_PACF_Plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for name in unique_names:
    region_data = data[data['name'] == name]

    if len(region_data) > 1:
        region_data.set_index('date', inplace=True)

        for feature in features:
            region_series = region_data[feature]

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'ACF and PACF for {name} - {feature}', fontsize=16)


            plot_acf(region_series, ax=axes[0], lags=20)
            axes[0].set_title(f'ACF of {name} - {feature}')


            plot_pacf(region_series, ax=axes[1], lags=20)
            axes[1].set_title(f'PACF of {name} - {feature}')


            plot_filename = f"{output_dir}/ACF_PACF_{name}_{feature}.png"
            plt.savefig(plot_filename)
            plt.close(fig)

print("ACF and PACF plots have been saved.")


