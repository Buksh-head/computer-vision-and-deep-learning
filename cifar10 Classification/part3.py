import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# simple neural network with one hidden layer
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # input -> hidden -> output
        self.fc1 = nn.Linear(3072, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten image
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# load data
transform = transforms.ToTensor()

train_full = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

train_data, val_data = random_split(train_full, [45000, 5000])


# optuna objective
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 128, 512, step=128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)

    model = NeuralNetwork(hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # short training for tuning
    for _ in range(5):
        model.train()
        for images, labels in train_loader:
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return evaluate(model, val_loader)


# run optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

print("Best params:", study.best_params)


# final model
model = NeuralNetwork(study.best_params["hidden_size"])
optimizer = optim.Adam(model.parameters(), lr=study.best_params["lr"])
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# full training
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Test Acc: {evaluate(model, test_loader)*100:.2f}%")

print("Final Test Accuracy:", evaluate(model, test_loader)*100)