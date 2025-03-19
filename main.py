import streamlit as st
import yfinance as yf
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Streamlit app layout
st.title("Prediksi Harga Cryptocurrency dengan LSTM")
st.write("Aplikasi ini memprediksi harga penutupan cryptocurrency menggunakan model LSTM.")

# Pilih aset crypto
asset_mapping = {'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD'}
asset_name_display = st.radio("Pilih Aset", options=list(asset_mapping.keys()))
asset = asset_mapping[asset_name_display]

# Pilih rentang tanggal
start_date = st.date_input("Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.date_input("Tanggal Akhir", pd.to_datetime("2024-01-01"))

# Pilih parameter model
time_step = st.radio("Time Step", options=[25, 50, 75, 100], index=3)
epoch_option = st.radio("Jumlah Epoch", options=[12, 25, 50, 100], index=1)

if st.button("Jalankan Prediksi"):
    if start_date >= end_date:
        st.error("Tanggal akhir harus lebih besar dari tanggal mulai.")
    else:
        st.write(f"Mengambil data harga {asset_name_display} dari Yahoo Finance...")
        df = yf.download(asset, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        
        if 'Date' in df.columns and 'Close' in df.columns:
            fig = px.line(df, x='Date', y='Close', title=f'Histori Harga Penutupan {asset_name_display}')
            st.plotly_chart(fig)
        else:
            st.error("Data yang diambil tidak memiliki kolom yang sesuai. Coba lagi dengan rentang tanggal yang berbeda.")

        # Preprocessing
        closedf = df[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

        # Split data
        training_size = int(len(closedf) * 0.90)
        train_data, test_data = closedf[:training_size], closedf[training_size:]

        # Function to create dataset
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Cek apakah jumlah data cukup untuk reshape
        if X_test.shape[0] == 0:
            st.error("Data test terlalu kecil, coba ubah rentang tanggal atau time step.")
        else:
            # Reshape data
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Build LSTM Model
            model = Sequential([
                LSTM(50, input_shape=(time_step, 1), activation="relu"),
                Dense(1)
            ])
            model.compile(loss="mean_squared_error", optimizer="adam")

            # Train Model
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_option, batch_size=32, verbose=1)

            # Predictions
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            # Inverse transform
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)
            original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
            original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Evaluation Metrics
            train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
            test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
            train_mape = np.mean(np.abs((original_ytrain - train_predict) / original_ytrain)) * 100
            test_mape = np.mean(np.abs((original_ytest - test_predict) / original_ytest)) * 100

            # Display metrics
            st.write("### Metrik Evaluasi")
            st.write(f"RMSE (Training): {train_rmse}")
            st.write(f"RMSE (Testing): {test_rmse}")
            st.write(f"MAPE (Training): {train_mape}%")
            st.write(f"MAPE (Testing): {test_mape}%")

            # Visualisasi prediksi
            plotdf = pd.DataFrame({
                'Date': df['Date'].values[time_step+1:len(train_predict)+len(test_predict)+time_step+1],
                'Original_Close': np.concatenate([original_ytrain.flatten(), original_ytest.flatten()]),
                'Predicted_Close': np.concatenate([train_predict.flatten(), test_predict.flatten()])
            })
            
            st.write(f"### Perbandingan Harga Penutupan Asli vs Prediksi untuk {asset_name_display}")
            fig = px.line(plotdf, x='Date', y=['Original_Close', 'Predicted_Close'],
                          labels={'value': 'Harga', 'Date': 'Tanggal'},
                          title=f'Harga Penutupan Asli vs Prediksi untuk {asset_name_display}')
            st.plotly_chart(fig)

            # Display hasil prediksi dalam DataFrame
            st.write("### Hasil Prediksi dalam DataFrame")
            st.write(plotdf)
