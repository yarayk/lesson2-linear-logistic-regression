import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_model(lr, batch_size, optimizer_name, epochs=100):
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LinearRegression(X_train.shape[1])
    criterion = nn.MSELoss()

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(dataloader))
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
    
    return losses, test_mse

learning_rates = [0.0001, 0.001, 0.01, 0.1]
batch_sizes = [8, 16, 32, 64, 128]
optimizers = ['SGD', 'Adam', 'RMSprop']
epochs = 100

results = []

# Эксперимент Скорость обучения
plt.figure(figsize=(12, 8))
for lr in learning_rates:
    losses, test_mse = train_model(lr, 32, 'Adam', epochs)
    plt.plot(losses, label=f'lr={lr}')
    results.append({
        'type': 'learning_rate',
        'param': lr,
        'test_mse': test_mse
    })
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.savefig('learning_rates.png')
plt.close()

# Эксперимент Размер батча
plt.figure(figsize=(12, 8))
for bs in batch_sizes:
    losses, test_mse = train_model(0.01, bs, 'Adam', epochs)
    plt.plot(losses, label=f'batch_size={bs}')
    results.append({
        'type': 'batch_size',
        'param': bs,
        'test_mse': test_mse
    })
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss for Different Batch Sizes')
plt.legend()
plt.grid(True)
plt.savefig('batch_sizes.png')
plt.close()

plt.figure(figsize=(12, 8))
for opt in optimizers:
    losses, test_mse = train_model(0.01, 32, opt, epochs)
    plt.plot(losses, label=opt)
    results.append({
        'type': 'optimizer',
        'param': opt,
        'test_mse': test_mse
    })
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss for Different Optimizers')
plt.legend()
plt.grid(True)
plt.savefig('optimizers.png')
plt.close()

import csv
with open('hyperparameter_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['type', 'param', 'test_mse'])
    writer.writeheader()
    writer.writerows(results)

# Результаты сохранены в hyperparameter_results.csv