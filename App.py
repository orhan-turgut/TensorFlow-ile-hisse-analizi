import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM

# İki Tarih Arasındaki Verilerimizi Çekiyoruz
kripto = 'btc-usd' # Yahoo Finance adresindeki kripto veya hisse kodu
start = dt.datetime(2020,1,1)
end = dt.datetime(2021,4,4)
veri = web.DataReader(kripto, 'yahoo', start, end)

tahmin_gunu = 60


x_train = []
y_train = []

for x in range(tahmin_gunu, len(scaled_data)):
    x_train.append(scaled_data[x - tahmin_gunu:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Sequential modelimizi oluşturuyoruz.
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Mevcut Veriler Üzerinde Test Modeli Doğruluğu'''
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()
test_data = web.DataReader(kripto, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((veri['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - tahmin_gunu:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Test verileri üzerinde tahmin yap
x_test = []
for x in range(tahmin_gunu, len(model_inputs)):
    x_test.append(model_inputs[x - tahmin_gunu:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Yapay Zeka Tahminini Grafiğe Döküyoruz
plt.plot(actual_prices, color="blue", label=f"{kripto} Şimdiki Fiyatı")
plt.plot(predicted_prices, color="green", label=f"{kripto} Tahmin Fiyatı")
plt.title(f"{kripto} Fiyat")
plt.xlabel('Zaman')
plt.ylabel('Fiyat')
plt.legend()
plt.show()
