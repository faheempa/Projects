import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

# get data
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species")
plt.show()

# organize data
data = torch.tensor(iris[iris.columns[0:4]].values, dtype=torch.float32)
labels = torch.zeros(len(data), dtype=torch.long)
labels[iris.species == "versicolor"] = 1
labels[iris.species == "virginica"] = 2

# train-test split
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# convert train and test data to PyTorch datasets
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

# create PyTorch dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

# ANN model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.input = nn.Linear(4, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.hidden1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.hidden2(x)
        return x

# Function to create and return the ANN model
def create_model():
    model = ANN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, loss_fn, optimizer

# Train the ANN model
def train_model(model, loss_fn, optimizer, train_loader, test_loader, epochs):
    train_acc = np.zeros(epochs)
    test_acc = np.zeros(epochs)
    for epoch in range(epochs):
        model.train()
        batch_accuracy = []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_accuracy.append(torch.sum(torch.argmax(outputs, dim=1) == labels).item() / len(labels))

        train_acc[epoch] = np.mean(batch_accuracy)*100
        model.eval()
        with torch.no_grad():
            x,y = next(iter(test_loader))
            outputs = model(x)
            test_acc[epoch] = torch.sum(torch.argmax(outputs, dim=1) == y).item() / len(y) * 100
        
        print(f"Epoch: {epoch+1:04}/{epochs}, Accuracy: {test_acc[epoch]:.2f}")
        
    return model, train_acc, test_acc

# Create and train the model
model, loss_fn, optimizer = create_model()
epochs = 1000
model, train_acc, test_acc = train_model(model, loss_fn, optimizer, train_loader, test_loader, epochs)

# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_acc, label="Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Final train accuracy: {:.2f}".format(train_acc[-1]))

plt.subplot(1, 2, 2)
plt.plot(range(epochs), test_acc, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Final test accuracy: {:.2f}".format(test_acc[-1]))   
plt.show()

# plot output with and without softmax
pred = model(data)
sm = nn.Softmax(dim=1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(pred.detach().numpy())
plt.legend(iris.species.unique())
plt.title("without softmax")

plt.subplot(1, 2, 2)
plt.plot(sm(pred).detach().numpy())
plt.legend(iris.species.unique())
plt.title("with softmax")
plt.show()

    


