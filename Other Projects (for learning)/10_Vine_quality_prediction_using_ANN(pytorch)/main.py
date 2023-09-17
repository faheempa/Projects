# imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn

# for data processing
import pandas as pd

# for numerical processing
import numpy as np
import scipy.stats as stats

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# read in data
df = pd.read_csv("data\winequality-red.csv", sep=';')
print(df.shape)

# remove outliers
df = df[df['total sulfur dioxide'] < 200]

cols2train = df.keys()
cols2train = cols2train.drop('quality')

# Data normalization zscore all cols except quality
for col in cols2train:
    df[col] = stats.zscore(df[col])

# converting to binary classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

# organise data
data = torch.tensor(df[cols2train].values).float()
labels = torch.tensor(df['quality'].values).float()
# making the labels 2D
labels = labels.unsqueeze(1)

# train test split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
print("train data shape: ", train_data.shape)
print("test data shape: ", test_data.shape)

# making train and test datasets
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

# making dataloaders
def create_dataloaders(batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    return train_loader, test_loader

# ANN model
class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, 32)
        self.bnorm32 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 64)
        self.bnorm64 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=self.bnorm32(self.relu(self.input(x)))
        x = self.bnorm64(self.relu(self.fc1(x)))
        x = self.bnorm32(self.relu(self.fc2(x)))
        x = self.output(x)
        return x

# test the model
model = ANN(input_dim=train_data.shape[1], output_dim=1)
print(model)

# create model
def create_model(input_dim, output_dim):
    model = ANN(input_dim, output_dim)
    optmizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    return model, optmizer, loss_fn

# train model
def train_model(model, optimizer, loss_fn, train_dataloader, test_dataloader, epochs=100):
    train_acc = np.zeros(epochs)
    test_acc = np.zeros(epochs)

    for epoch in range(epochs):
        # training
        model.train()
        batch_acc = []
        for inputs, labels in train_dataloader:
            preds = model(inputs)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_acc.append(100*torch.mean(((preds > 0) == labels).float()).item())

        # training accuracy
        train_acc[epoch] = np.mean(batch_acc)

        # testing
        model.eval()
        with torch.no_grad():
            inputs, labels = next(iter(test_dataloader))
            preds = model(inputs)
            test_acc[epoch] = 100*torch.mean(((preds > 0) == labels).float()).item()

        if epoch % 200 == 0:
            print(f"completed {epoch} epochs.")

    return train_acc, test_acc

epochs = 1000
train_dataloader, test_dataloader = create_dataloaders(batch_size=32)
model, optimizer, loss_fn = create_model(input_dim=train_data.shape[1], output_dim=1)
train_acc_2, test_acc_2 = train_model(model, optimizer, loss_fn, train_dataloader, test_dataloader, epochs)

# plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc_2)
plt.title("Train Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# plot results
plt.subplot(1, 2, 2)
plt.plot(test_acc_2)
plt.title("Test Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()