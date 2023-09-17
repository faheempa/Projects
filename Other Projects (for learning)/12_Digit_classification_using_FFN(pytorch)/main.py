# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# import dataset (comes with colab!)
data = np.loadtxt(open("mnist_train_small.csv", "rb"), delimiter=",")

# shape of the data matrix
print(data.shape)

# extract labels (number IDs) and remove from data
labels = data[:, 0]
data = data[:, 1:]

print(labels.shape)
print(data.shape)


def show_digit_pic(index, data, labels):
    n = 0
    while n < len(index):
        plt.figure(figsize=(12, 6))
        for i in range(min(10, len(index) - n)):
            plt.subplot(2, 5, i + 1)
            idx = n + i
            img = np.reshape(data[index[idx], :], (28, 28))
            plt.imshow(img, cmap="gray")
            plt.title("The number %i" % labels[index[idx]])
        plt.show()
        n += 10


def digit_in_fnn_vision(index, data, labels):
    n = 0
    while n < len(index):
        fig, axs = plt.subplots(2, 4, figsize=(12, 6))
        for ax in axs.flatten():
            # create the image
            if n == len(index):
                break
            ax.plot(data[index[n], :], "ko")
            ax.set_title("The number %i" % labels[index[n]])
            n += 1
        plt.show()


# normalize data
dataNorm = data / np.max(data)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(data.flatten())
ax[0].set_xlabel("Normalized pixel values")
ax[0].set_ylabel("Counts")
ax[0].set_title("Histogram of original data")
ax[1].hist(dataNorm.flatten())
ax[1].set_xlabel("Normalized pixel values")
ax[1].set_ylabel("Counts")
ax[1].set_title("Histogram of normalized data")
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')
plt.show()

# convert to tensor
dataTensor = torch.tensor(dataNorm, dtype=torch.float)
labelsTensor = torch.tensor(labels, dtype=torch.long)

# split into training and test set
train_data, test_data, train_labels, test_labels = train_test_split(
    dataTensor, labelsTensor, test_size=0.2, random_state=42
)

# convert to torch dataset and to dataloader
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

print("Number of training samples: %i" % len(train_dataset))
print("Number of test samples: %i" % len(test_dataset))
print("Number of batches: %i" % len(train_loader))
print("Shape of one batch: %s" % str(next(iter(train_loader))[0].shape))


# create a simple feedforward neural network
def create_model():
    class ANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(784, 128)
            self.hidden1 = nn.Linear(128, 64)
            self.hidden2 = nn.Linear(64, 32)
            self.output = nn.Linear(32, 10)
            self.norm128 = nn.BatchNorm1d(128)
            self.norm264 = nn.BatchNorm1d(64)
            self.norm232 = nn.BatchNorm1d(32)
            self.dr = 0.5

        def forward(self, x):
            x = F.relu(self.input(x))
            x = self.norm128(x)
            x = F.dropout(x, p=self.dr, training=self.training)
            x = F.relu(self.hidden1(x))
            x = self.norm264(x)
            x = F.dropout(x, p=self.dr, training=self.training)
            x = F.relu(self.hidden2(x))
            x = self.norm232(x)
            x = F.dropout(x, p=self.dr, training=self.training)
            x = F.log_softmax(self.output(x), dim=1)
            return x

    model = ANN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    return model, optimizer, criterion


# test the model
# test_model, test_optimizer, test_criterion = create_model()
# test_input = torch.randn(64, 784)
# test_output = test_model(test_input)
# print("Shape of input: %s" % str(test_input.shape))
# print(torch.exp(test_output))  # output is log-probabilities, use exp to get probabilities


def train_model(train_loader, test_loader):
    epochs = 100
    model, optimizer, criterion = create_model()
    losses = np.zeros(epochs)
    trainAcc = np.zeros(epochs)
    testAcc = np.zeros(epochs)

    for i in range(epochs):
        model.train()
        batchAcc = []
        bactchLoss = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batchAcc.append(
                100 * torch.mean((labels == torch.argmax(outputs, dim=1)).float())
            )
            bactchLoss.append(loss.item())
        losses[i] = np.mean(bactchLoss)
        trainAcc[i] = np.mean(batchAcc)

        # test model
        model.eval()
        x, y = next(iter(test_loader))
        with torch.no_grad():
            outputs = model(x)
            testAcc[i] = 100 * torch.mean((y == torch.argmax(outputs, dim=1)).float())
        print(f"Epoch: {i+1}/{epochs}, Train acc: {trainAcc[i]:.2f}")

    return model, losses, trainAcc, testAcc


# train a model
model, losses, trainAcc, testAcc = train_model(train_loader, test_loader)
# plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(trainAcc, label="Train accuracy")
plt.plot(testAcc, label="Test accuracy")
plt.legend()
plt.title("Training results")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# plot some sample
random_indexs = np.random.randint(0, len(test_dataset), 4)
x, y = test_dataset[random_indexs]
prediction = model(x).detach()

plt.figure(figsize=(7, 7))
for i, pred in enumerate(prediction):
    plt.subplot(2, 2, i + 1)
    plt.bar(range(10), torch.exp(pred))
    plt.xticks(np.arange(10))
    plt.title(f"True value: {y[i]}")
plt.show()

# plot the error values
x, y = next(iter(test_loader))
yhat = model(x).detach()
prediction = torch.argmax(torch.exp(yhat), dim=1)
missed_index = np.array(np.where(prediction != y))[0][:10]
show_digit_pic(missed_index, x, prediction)

plt.figure(figsize=(15, 12))
for i, idx in enumerate(missed_index[1:]):
    plt.subplot(3,3,i+1)
    plt.bar(range(10), torch.exp(yhat[idx]))
    plt.xticks(np.arange(10))
    plt.title(f"pred value: {prediction[idx]}, true value: {y[idx]}");

plt.show()