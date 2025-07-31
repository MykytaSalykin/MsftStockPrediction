# %% [Imports]
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import logging
import sys

# Make sure we're using a compatible Python version (I had this issue)
if sys.version_info >= (3, 14):
    logging.warning(
        "Python 3.14 may not be fully supported by PyTorch. Consider using Python 3.12."
    )

# Set up logging to track progress
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Check for MPS (Apple Silicon GPU) or fall back to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logging.info(f"Using device: {device}")
# %% [Data Download and Preprocessing]
try:
    # Pull MSFT stock data starting from July 2023
    ticker = "MSFT"
    logging.info(f"Fetching data for {ticker}")
    df = yf.download(ticker, start="2023-07-07")
    if df.empty:
        raise ValueError("No data downloaded from yfinance")

    # Scale closing prices for better model training
    scaler = StandardScaler()
    df["Close"] = scaler.fit_transform(df[["Close"]]).flatten()

    sequence_length = 30
    data = []
    for i in range(len(df) - sequence_length):
        data.append(df["Close"][i : i + sequence_length])
    data = np.array(data)  # Shape: (num_sequences, sequence_length)
    data = data[:, :, np.newaxis]  # Shape: (num_sequences, sequence_length, 1)
    logging.info(f"Data shape after preprocessing:  {data.shape}")

    training_size = int(0.7 * len(data))
    X_train = (
        torch.from_numpy(data[:training_size, :-1, :])
        .squeeze(-1)
        .type(torch.FloatTensor)
        .to(device)
    )  # Changed: squeeze(-1) and FloatTensor
    Y_train = (
        torch.from_numpy(data[:training_size, -1, :])
        .squeeze(-1)
        .type(torch.FloatTensor)
        .to(device)
    )  # Changed: squeeze(-1) and FloatTensor
    X_test = (
        torch.from_numpy(data[training_size:, :-1, :])
        .squeeze(-1)
        .type(torch.FloatTensor)
        .to(device)
    )  # Changed: squeeze(-1) and FloatTensor
    Y_test = (
        torch.from_numpy(data[training_size:, -1, :])
        .squeeze(-1)
        .type(torch.FloatTensor)
        .to(device)
    )  # Changed: squeeze(-1) and FloatTensor

    # Log shapes to confirm everything's set
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"Y_train shape: {Y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"Y_test shape: {Y_test.shape}")

except Exception as e:
    logging.error(f"Data prep failed: {e}")
    raise


# %% [Model Definition]
# I'll be using LSTM model cause it's good for time series data
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        # Run through LSTM and grab the last output
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


model = Model(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
logging.info("Model initialized")

# %% [Training Setup]
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 200

# %% [Training Loop]
logging.info("Starting training...")
try:
    for i in range(num_epochs):
        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, Y_train)
        if i % 25 == 0:
            logging.info(f"Epoch {i}, Loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info("Training complete!")
except Exception as e:
    logging.error(f"Training failed: {e}")
    raise
# %% [Evaluation]
# Evaluation part
model.eval()
try:
    with torch.no_grad():
        logging.info(
            f"X_test shape before model: {X_test.shape}"
        )  # Added for debugging
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, Y_test)
        logging.info(f"Test Loss: {test_loss.item()}")
        rmse = root_mean_squared_error(Y_test.cpu().numpy(), y_test_pred.cpu().numpy())
        logging.info(f"Test RMSE: {rmse}")
except Exception as e:
    logging.error(f"Evaluation failed: {e}")
    raise

# %% [Visualization]
# Plot actual vs predicted prices (inversing for interpretability)
try:
    y_test_pred_np = scaler.inverse_transform(y_test_pred.cpu().numpy())
    Y_test_np = scaler.inverse_transform(Y_test.cpu().numpy())

    plt.figure(figsize=(10, 6))
    plt.plot(Y_test_np, label="Actual Prices", color="blue")
    plt.plot(y_test_pred_np, label="Predicted Prices", color="orange")
    plt.title("MSFT Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()
except Exception as e:
    logging.error(f"Plotting failed: {e}")
    raise
