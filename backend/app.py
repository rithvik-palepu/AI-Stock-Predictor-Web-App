from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM

app = Flask(__name__)
CORS(app)

def fetch_and_predict(stock_ticker): 
  stock_data = yf.download(stock_ticker, start = '2014-03-08', end = '2024-03-08')
  print(stock_data)

  def calculate_investment_decision(data, future_days=5, threshold=0.02):
    future_returns = data['Close'].shift(-future_days) / data['Close'] - 1
    investment_decision = (future_returns > threshold).astype(int)
    return investment_decision

  stock_data['Invest'] = calculate_investment_decision(stock_data)

  scaler = MinMaxScaler(feature_range = (0,1))

  scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))

  def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
      X.append(data[i:(i + time_step), 0])
      y.append(data[i + time_step, 0])

    return np.array(X), np.array(y)

  time_step = 100

  X, y = create_dataset(scaled_data, time_step)

  train_size = 0.8

  X_train, x_test = X[:int(X.shape[0] * train_size):], X[int(X.shape[0] * train_size):]
  y_train, y_test = y[:int(y.shape[0] * train_size):], y[int(y.shape[0] * train_size):]

  model = Sequential()
  model.add(LSTM(64, input_shape = (time_step, 1)))
  model.add(Dense(64))
  model.add(Dense(64))
  model.add(Dense(1))

  model.compile(optimizer = 'adam', loss = "mean_squared_error", metrics = ['accuracy'])
  model.fit(X_train, y_train, epochs = 10, batch_size = 64)

  test_loss = model.evaluate(x_test, y_test)
  print(test_loss)

  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  original_data = stock_data['Close'].values
  predicted_data = np.empty_like(original_data, dtype = np.float64)
  predicted_data[:] = np.nan
  predicted_data[-len(predictions):] = predictions.reshape(-1, 1)

  new_predictions = model.predict(x_test[-90:])
  new_predictions = scaler.inverse_transform(new_predictions)

  predicted_data = np.append(predicted_data, new_predictions)

  def save_plot(original_data, predicted_data, stock_ticker):
    # Get the absolute path to the 'backend/plots' directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    plot_dir = os.path.join(script_dir, "plots")  # Ensure we save inside 'backend/plots'

    # Ensure the 'backend/plots' directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_path = os.path.join(plot_dir, f"{stock_ticker}_prediction.png")
    print(f"Saving plot to: {plot_path}")

    # Generate and save the plot
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(original_data, label="Actual Prices")
        plt.plot(predicted_data, label="Predicted Prices", linestyle='dashed')
        plt.legend()
        plt.title(f"Stock Price Prediction for {stock_ticker}")
        plt.savefig(plot_path)
        print(f"Plot saved successfully to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()

    return plot_path
  plot_path = save_plot(original_data, predicted_data, stock_ticker)
  return plot_path

@app.route('/')
def home():
    return "Welcome to the Stock Prediction Web App"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data: ", data)

    stock_ticker = data.get("ticker", "").upper()

    if not stock_ticker:
        return jsonify({"error": "Stock ticker is required"}), 400

    try:
        plot_path = fetch_and_predict(stock_ticker)
        print(f"Sending plot at: {plot_path}")
        return send_file(plot_path, mimetype='image/png')
    except Exception as e:
        print("Error occurred:", str(e)) 
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)