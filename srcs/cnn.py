import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm


class EarlyStopping:
    def __init__(self, patience=2, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss: float) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


class CNN(nn.Module):

    def __init__(self, dataset=None, epochs=20, batch_size=64, lr=0.001):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.classes = dataset.classes if dataset is not None else None
        self.train_loss = []
        self.val_loss = []

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 8)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else torch.backends.mps.is_available()
            and torch.device("mps")
            or torch.device("cpu")
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=5, delta=0.001, verbose=True)

    def save_model(self) -> None:
        model_directory = os.path.join(os.getcwd(), "models")
        if not os.path.exists(model_directory):
            os.makedirs(model_directory, exist_ok=True)
        path = os.path.join(model_directory, f"model{self.epochs}.zip")
        torch.save(
            {"state_dict": self.state_dict(), "classes": self.classes},
            path,
        )
        print(
            f"Model saved to {os.path.join(model_directory, f'model{self.epochs}.zip')}"
        )

    def display_loss(self) -> None:
        plt.plot(self.train_loss, label="Training Loss")
        plt.plot(self.val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        print("Starting training...")

        for epoch in tqdm.tqdm(range(self.epochs), desc="Training"):

            # --- Training ---
            self.train()
            epoch_train_loss = 0.0

            for _, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self(images)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()

            self.train_loss.append(epoch_train_loss / len(train_loader))

            # --- Validation ---
            self.eval()
            with torch.no_grad():
                epoch_val_loss = 0.0

                for _, (images, labels) in enumerate(test_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self(images)
                    loss = self.criterion(outputs, labels)

                    epoch_val_loss += loss.item()

                self.val_loss.append(epoch_val_loss / len(test_loader))

            self.early_stopping.check_early_stop(self.val_loss[-1])
            if self.early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch}")
                break

    def test(self, test_loader: DataLoader) -> None:
        self.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(8)]
            n_class_samples = [0 for i in range(8)]
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)

                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(self.batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f"Accuracy of the network: {acc} %")

            for i in range(len(labels)):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f"Accuracy of {self.classes[i]}: {acc} %")

    def predict(self, image: torch.Tensor, classes: list) -> str:
        self.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self(image.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            return classes[predicted.item()]
