import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
from sklearn.metrics import recall_score, precision_score

# Set GPU device
print(torch.cuda.is_available())
device = torch.device("cuda:0")

# %% Load data
TRAIN_ROOT = "/content/drive/MyDrive/Colab_Notebooks/ML/data/training"
TEST_ROOT = "/content/drive/MyDrive/Colab_Notebooks/ML/data/testing"
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_ROOT)
test_dataset = torchvision.datasets.ImageFolder(root=TRAIN_ROOT)


# %% Building the model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)

        # Replace output layer according to our problem
        in_feats = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_feats, 2)

    def forward(self, x):
        x = self.vgg16(x)
        return x


model = CNNModel()
model.to(device)
model

# %% Prepare data for pretrained model
train_dataset = torchvision.datasets.ImageFolder(
    root=TRAIN_ROOT,
    transform=transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor()
    ])
)

test_dataset = torchvision.datasets.ImageFolder(
    root=TEST_ROOT,
    transform=transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor()
    ])
)

# train_dataset[0][0].permute(1,2,0)

# %% Create data loaders
batch_size = 31
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Train
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
epochs = 10

# Lists to store losses
losses = []

for epoch in range(epochs):
    for i, batch in enumerate(train_loader, 0):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cross_entropy_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collect loss
        losses.append(loss.item())

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Plot loss over iterations
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Loss During Training')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 测试模型在测试集上的性能
model.eval()  # 将模型设置为评估模式
test_labels = []
test_preds = []

with torch.no_grad():  # 在评估过程中不计算梯度
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(predicted.cpu().numpy())

# 计算测试集上的宏观召回率、宏观精确度以及准确度
test_recall = recall_score(test_labels, test_preds, average='macro')
test_precision = precision_score(test_labels, test_preds, average='macro')
test_accuracy = np.mean(np.array(test_labels) == np.array(test_preds))

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test Precision: {test_precision:.4f}')
