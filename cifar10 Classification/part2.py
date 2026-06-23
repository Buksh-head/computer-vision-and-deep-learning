import numpy as np
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
import torch
import torch.nn as nn
import torch.optim as optim


class SoftmaxClassifier:
    def __init__(self, learning_rate, num_classes, num_features):
        self.learning_rate = learning_rate
        # linear layer maps input features to 10 class scores
        self.model = nn.Linear(num_features, num_classes)

        # cross entropy applies softmax internally during training
        self.criterion = nn.CrossEntropyLoss()

        # SGD updates the weights using gradients
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
    def train(self, X, y, epochs, print_every):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        for epoch in range(epochs):
            # Forward pass: logits (no softmax here)
            logits = self.model(X_tensor)
            loss = self.criterion(logits, y_tensor)

            # Backward pass + gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % print_every == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d} - Loss: {loss.item():.4f}")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1)
        return preds.numpy()
    
def dataset_to_numpy(dataset):
    X = []
    y = []

    for img, label in dataset:
        X.append(img.numpy().reshape(-1))  # flatten
        y.append(label)

    return np.array(X), np.array(y)

def extract_lbp_features(X, P=8, R=1):
    features = []

    for img in X:
        # convert RGB image to grayscale
        gray = rgb2gray(img)
        gray = (gray * 255).astype(np.uint8)
        
        # compute LBP image
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")

        # histogram of LBP values
        n_bins = P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        # normalise histogram
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)

        features.append(hist)

    return np.array(features)

def dataset_to_numpy_images(dataset):
    X = []
    y = []

    for img, label in dataset:
        # convert tensor [C, H, W] to numpy [H, W, C]
        X.append(img.numpy().transpose(1, 2, 0))
        y.append(label)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    transform = transforms.ToTensor()

    train_data = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    X_train, y_train = dataset_to_numpy(train_data)
    X_test, y_test = dataset_to_numpy(test_data)

    # Initialize and train classifier
    classifier = SoftmaxClassifier(learning_rate=0.01, num_classes=10, num_features=3072)
    classifier.train(X_train, y_train, epochs=900, print_every=90)

    # Predict on test set
    predictions = classifier.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    print("Part 2(a) - Softmax (raw pixels)")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    X_train_img, y_train = dataset_to_numpy_images(train_data)
    X_test_img, y_test = dataset_to_numpy_images(test_data)

    X_train_feat = extract_lbp_features(X_train_img)
    X_test_feat = extract_lbp_features(X_test_img)

    # initialise and train classifier on LBP features
    classifier = SoftmaxClassifier(
        learning_rate=10,
        num_classes=10,
        num_features=X_train_feat.shape[1]
    )
    classifier.train(X_train_feat, y_train, epochs=1800, print_every=360)

    # predict on test set
    predictions = classifier.predict(X_test_feat)
    accuracy = np.mean(predictions == y_test)

    print("Part 2(b) - Softmax (LBP features)")
    print(f"Test accuracy: {accuracy * 100:.2f}%")