from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import functional as F
from torch import nn
from numpy.random import seed
from datetime import datetime
import yfinance as yf
import pandas_datareader.data as pdr
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sqlite3
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

plt.style.use('seaborn')
yf.pdr_override()

# For reproducability
seed(1)
tf.random.set_seed(2)


# Some functions to help out with
def plot_predictions(test, predicted, symbol):
    plt.plot(test, color='red', label=f'Real {symbol} Stock Price')
    plt.plot(predicted, color='blue', label=f'Predicted {symbol} Stock Price')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{symbol} Stock Price')
    plt.legend()
    plt.show()


def plot_return_predictions(test, predicted, symbol):
    plt.plot(test, color='red', label=f'Real {symbol} Stock Price Returns')
    plt.plot(predicted, color='blue',
             label=f'Predicted {symbol} Stock Price Return')
    plt.title(f'{symbol} Stock Return Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{symbol} Stock Price Returns')
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    return rmse


def get_ticker_data(ticker: str, param_start_date, param_end_date) -> tuple:
    # Fetch data from Yahoo Finance
    data = pdr.get_data_yahoo(
        ticker, start=param_start_date, end=param_end_date)
    data.reset_index(inplace=True)
    data = data.fillna(method="ffill", axis=0)
    data = data.fillna(method="bfill", axis=0)

    # Prepare the data
    prices = data["Close"].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    prices = scaler.fit_transform(prices)

    # Create sequences
    SEQ_LEN = 8  # I assumed the sequence length, you can change this value as per your requirements
    inputs, targets = [], []
    for i in range(SEQ_LEN, len(prices) - 1):
        inputs.append(prices[i - SEQ_LEN:i])
        targets.append(prices[i + 1])

    return data, np.array(inputs), np.array(targets)


def fetch_ticker_data(ticker: str, start_date, end_date) -> tuple:
    return get_ticker_data(ticker, start_date, end_date)


def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e

# Scaled Exponentially-Regularized Linear Unit to try out - if anyone can make this work, let me know.

# def serlu(x, lambdaa=1.07862, alphaa=2.90427):
#     result = tf.cond(x >= 0, lambda: tf.math.multiply(lambdaa, x), lambda: tf.math.multiply(lambdaa, alphaa, x, tf.exp(x)))
#     return result
# Example usage:


symbol_to_fetch = 'IBM'
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
stock, inputs, targets = fetch_ticker_data(
    symbol_to_fetch, start_date, end_date)
stock.columns = ['DateTime', 'High', 'Low',
                 'Open', 'Close', 'Volume', 'Adj Close']
stock['DateTime'] = pd.to_datetime(stock['DateTime'])
stock.set_index('DateTime', inplace=True)
stock['Symbol'] = symbol_to_fetch

# save a copy for later testing
original_stock = stock
original_symbol = symbol_to_fetch

# print(stock_df.tail())
# print(inputs[-5:], targets[-5:])
# Choose a stock symbol
symbol_to_fetch = 'IBM'
# Choose a date range
start_date = str(datetime(2017, 1, 1).date())
end_date = str(datetime(2021, 2, 18).date())
# end_date = datetime.now().strftime('%Y-%m-%d')

# We have chosen the target as 'Close' attribute for prices. Let's see what it looks like
target = 'Close'  # this is accessed by .iloc[:,3:4].values below
train_start_date = start_date
train_end_date = '2021-10-31'
test_start_date = '2021-11-01'
training_set = stock[train_start_date:train_end_date].iloc[:, 3:4].values
test_set = stock[test_start_date:].iloc[:, 3:4].values

test_set_return = stock[test_start_date:].iloc[:, 3:4].pct_change().values
# log_return_test = np.log(test_set_return)

print(training_set.shape)
print(test_set.shape)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, head_size, num_heads, ff_dim, dropout=0):
        super(TransformerEncoder, self).__init__()
        print(
            f'TransformerEncoder input dim: {input_dim}, head size: {head_size}, num heads: {num_heads}, ff dim: {ff_dim}, dropout: {dropout}')
        # Get from 8 to head_size * num_heads)
        # Normalize on (head_size * num_heads, 1, 8)
        # self.ln1 = nn.LayerNorm(input_dim, eps=1e-6)
        # Get from (485, x, head_size * num_heads) to (485, x, head_size * num_heads)
        self.attn = nn.MultiheadAttention(
            embed_dim=head_size*num_heads, num_heads=num_heads, dropout=dropout, batch_first=True)
        # New dimension
        # new_dim = (input_dim[0], input_dim[1], head_size * num_heads)
        self.dropout1 = nn.Dropout(dropout)
        # Don't put the batch size for the dimensions for normalization
        self.ln2 = nn.LayerNorm(input_dim[1:], eps=1e-6)
        # Equivalent to Conv1D with kernel size 1 in TF
        # Get from head_size * num_heads to ff_dim
        self.conv1 = nn.Linear(head_size * num_heads, ff_dim)
        self.dropout2 = nn.Dropout(dropout)
        # Equivalent to Conv1D with kernel size 1 in TF
        self.conv2 = nn.Linear(ff_dim, head_size*num_heads)
        # self.ln3 = nn.LayerNorm(new_dim, eps=1e-6)

    def forward(self, inputs):
        print(f"forward Input shape: {inputs.shape}")
        # Apply the new embedding layer first
        # x = self.embedding(inputs)
        # print(f"forward x shape: {x.shape}")

        # Input shape: (batch_size, seq_len, input_dim)
        # x = self.ln1(inputs)
        # print(f"Shape after ln1: {x.shape}")
        # x is reshaped to (seq_len, batch_size, input_dim) because PyTorch's MultiheadAttention requires this format
        # x = x.permute(1, 0, 2)
        # print(f"Shape after permute: {x.shape}")

        # (batch_size, input_dim, seq_len)
        # x, _ = self.attn(inputs, inputs, inputs, is_causal=True)  # add mask
        x, _ = self.attn(inputs, inputs, inputs)
        print(f"Shape after attn: {x.shape}")
        # reshaping back to (batch_size, seq_len, input_dim)
        # x = self.dropout1(x.permute(1, 0, 2))
        x = self.dropout1(x)
        print(f"Shape after dropout: {x.shape}")
        # reduce back to input_dim[2]
        # res = x + inputs
        res = x + inputs
        print(f"Shape after res: {x.shape}")
        layer_norm = self.ln2(res)
        print(f"Shape after ln2: {layer_norm.shape}")
        x = F.relu(self.conv1(layer_norm))
        print(f"Shape after relu: {x.shape}")
        x = self.dropout2(x)
        print(f"Shape after dropout: {x.shape}")
        x = self.conv2(x)
        print(f"Shape after feed forward: {x.shape}")

        # return x + layer_norm
        return x + res


class Model(nn.Module):
    def __init__(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
        super(Model, self).__init__()
        # (batch_size, seq_len, input_dim)
        # (485, 8, 1)
        # self.input_shape = input_shape
        # Turn input shape tensor into a tuple
        self.input_shape = tuple(input_shape)

        # Each price is being embedded as a vector of size head_size * num_heads
        self.embedding = nn.Linear(
            self.input_shape[2], head_size * num_heads)  # New embedding layer
        print(f'Model input shape: {self.input_shape}, head size: {head_size}, num heads: {num_heads}, ff dim: {ff_dim}, num transformer blocks: {num_transformer_blocks}, mlp units: {mlp_units}, dropout: {dropout}, mlp dropout: {mlp_dropout}')
        # Dimension after embedding layer
        new_dim = (self.input_shape[0],
                   self.input_shape[1], head_size * num_heads)
        self.transformer_blocks = nn.ModuleList([TransformerEncoder(
            new_dim, head_size, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)])
        # Reduction from embeddings layer back to 1
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        # self.conv1 = nn.Linear(
        #     head_size * num_heads, self.input_shape[2])  # New embedding layer
        self.mlp = nn.Sequential(
            *[nn.Sequential(nn.Linear(self.input_shape[1], dim), nn.ELU(),
                            nn.Dropout(mlp_dropout)) for dim in mlp_units],
            nn.Linear(mlp_units[-1], 1)
        )

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, input_dim)
        print(f"model forward Input shape: {inputs.shape}")
        embeddings = self.embedding(inputs)
        # (batch_size, seq_len, head_size * num_heads)
        print(f"Shape after embedding: {embeddings.shape}")
        # Input dimensions
        x = embeddings
        for i, transformer_block in enumerate(self.transformer_blocks):
            x = transformer_block(x)
            print(f"Shape after transformer block {i+1}: {x.shape}")
        # x = self.conv1(x)
        # print(f"Shape after conv1: {x.shape}")
        # # GlobalAveragePooling in PyTorch is done with AdaptiveAvgPool1D
        # # permute and squeeze are used to get the right shape
        x = self.pool(x).squeeze(-1)
        print(f"Shape after pooling: {x.shape}")
        x = self.mlp(x)
        print(f"Final output shape: {x.shape}")
        # inputs: (batch_size, seq_len, input_dim)
        return x


def lr_scheduler(epoch, lr, warmup_epochs=30, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr


# Scaling the training set - I've tried it without scaling and results are very poor.
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

timesteps = 8
# First, we create data sets where each sample has with 8 timesteps and 1 output
# So for each element of training set, we have 8 previous training set elements
x_train = []
y_train = []

# subtract 1 because y_train will need one more value
for i in range(timesteps, training_set.shape[0] - 1):
    x_train.append(training_set_scaled[i-timesteps:i, 0])
    # start from i-timesteps+1 to shift one step
    y_train.append(training_set_scaled[i+1, 0])

# Notice how the first y_train value becomes the last X_train value for the next sample
x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train[0], y_train[0])
print(x_train[1], y_train[1])

# Notice how the first y_train value becomes the last X_train value for the next sample
# Notice how the first y_train value becomes the last X_train value for the next sample
# print(x_train.shape, y_train.shape)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
y_train = y_train.reshape((y_train.shape[0], 1))
# x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
print(x_train.shape, y_train.shape)

print(x_train.shape, y_train.shape, type(x_train), type(y_train))
# Interestingly - randomly arranging the samples works well, since we are using validation_split = 0.2, (rather then validation_data = )
# It is worth looking into whether using a K-fold would work better - if so would not use random permutation.
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# Do the same for x_test and y_test
x_test = []
y_test = []
testing_set_scaled = sc.transform(test_set)
for i in range(timesteps, testing_set_scaled.shape[0] - 1):
    x_test.append(testing_set_scaled[i-timesteps:i, 0])
    # y_test.append(testing_set_scaled[i-timesteps+1:i+1, 0])
    y_test.append(testing_set_scaled[i+1, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
y_test = y_test.reshape((y_test.shape[0], 1))
print(x_test.shape, y_test.shape)

# Following is your training part. Make sure your x_train and y_train are torch tensors
# and are on the right device (cpu or gpu) before passing into the model.
# assuming x_train is numpy array
mps_device = torch.device("mps")
x_train = torch.tensor(x_train, device=mps_device, dtype=torch.float32)
# assuming y_train is numpy array
y_train = torch.tensor(y_train, device=mps_device, dtype=torch.float32)
y_test = torch.tensor(y_test, device=mps_device, dtype=torch.float32)
x_test = torch.tensor(x_test, device=mps_device, dtype=torch.float32)

# Create a TensorDataset
train_data = TensorDataset(x_train, y_train)

# Create a TensorDataset for the test data
print(x_test.shape, y_test.shape)
test_data = TensorDataset(x_test, y_test)

# Define batch size
batch_size = 32

# Create a DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Create a DataLoader for the test data
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = Model(
    input_shape=x_train.shape,
    head_size=46,
    num_heads=60,
    ff_dim=55,
    num_transformer_blocks=5,
    mlp_units=[256],
    mlp_dropout=0.4,
    dropout=0.14,
).to(mps_device)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

scheduler = LambdaLR(optimizer, lambda epoch: lr_scheduler(
    epoch, optimizer.param_groups[0]['lr']))


tb = SummaryWriter()

# In your training loop, iterate over train_loader to get batches of data
epochs = 100  # number of epochs
for epoch in range(epochs):
    for i, (x_batch, y_batch) in enumerate(train_loader):
        model.train()  # switch to training mode
        optimizer.zero_grad()  # reset gradients

        output = model(x_batch)
        loss = criterion(output, y_batch)

        loss.backward()  # compute gradients
        optimizer.step()  # update weights

        scheduler.step()  # update learning rate

        # Record the loss into the tensorboard
        tb.add_scalar("Loss", loss.item(), epoch*len(train_loader) + i)

        # optionally print loss here
        if i % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

    # Evaluation phase
    # with torch.no_grad():
    #     total_test_loss = 0
    #     for i, (x_test_batch, y_test_batch) in enumerate(test_loader):
    #         model.eval()
    #         test_output = model(x_test_batch)
    #         test_loss = criterion(test_output, y_test_batch)
    #         total_test_loss += test_loss.item()

    #     avg_test_loss = total_test_loss / len(test_loader)
    #     tb.add_scalar("Test Loss", avg_test_loss, epoch)
    #     print(f"Epoch: {epoch}, Test Loss: {avg_test_loss}")


tb.flush()
