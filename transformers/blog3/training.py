
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas_datareader as pdr
import yfinance as yf
from blog3.models import ModelTrunk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def lr_lambda(epoch, warmup_epochs=15, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr / base_lr


yf.pdr_override()

# Parameters
BATCH_SIZE = 64
N_EPOCHS = 100
SEQ_LEN = 60  # The number of past days to consider
MODEL_DIM = 128  # Dimensionality of the model
NUM_HEADS = 2  # Number of attention heads
NUM_LAYERS = 1  # Number of attention layers
DROPOUT = 0.1  # Dropout rate

# Prepare the data
data = pdr.get_data_yahoo("AAPL", start="2015-01-01", end="2021-09-01")
prices = data["Close"].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
prices = scaler.fit_transform(prices)

# Create sequences
inputs = []
targets = []

for i in range(SEQ_LEN, len(prices)):
    inputs.append(prices[i - SEQ_LEN:i])
    targets.append(prices[i])

inputs = torch.tensor(inputs).float()
targets = torch.tensor(targets).float()

# Train/test split
train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    inputs, targets, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(test_inputs, test_targets)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model
model = ModelTrunk(time2vec_dim=MODEL_DIM, num_heads=NUM_HEADS, head_size=MODEL_DIM,
                   ff_dim=MODEL_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)

# Initialize the optimizer and scheduler
optimizer = optim.AdamW(model.parameters())
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Train the model
for epoch in range(N_EPOCHS):
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs)
        loss = F.mse_loss(outputs, batch_targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    scheduler.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{N_EPOCHS}, Loss: {loss.item()}')
