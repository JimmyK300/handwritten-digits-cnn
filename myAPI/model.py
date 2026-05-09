import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
from torchvision import transforms

from myAPI.mnist_to_pt import mnist_pt

from pathlib import Path
from tqdm import trange

import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class MNISTDataset(Dataset):
    def __init__(self, path, train=True, transform = None, target_transform = None):
        if train:
            img_file = Path(path) / "train-images-idx3-ubyte"
            lbl_file = Path(path) / "train-labels-idx1-ubyte"
        else:
            img_file = Path(path) / "t10k-images-idx3-ubyte"
            lbl_file = Path(path) / "t10k-labels-idx1-ubyte"
            
        # Use your custom function to read the raw bytes
        self.images = mnist_pt(img_file)
        self.labels = mnist_pt(lbl_file)
        
        # Normalize to [0, 1] and add the channel dimension (N, 1, 28, 28)
        self.images = self.images.to(torch.float32) / 255.0
        self.images = self.images.unsqueeze(1)
        
        # Ensure labels are standard long integers for loss functions (like CrossEntropy)
        self.labels = self.labels.to(torch.long)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class MyData(Dataset):
    def __init__(self, path, train=True, transform = None, target_transform = None):
        TrainFileNames = ["train_images.bin", "train_labels.bin",]
        TestFileNames  = ["test_images.bin",  "test_labels.bin",]

        if train:
            images_file = path + '\\' + TrainFileNames[0]
            labels_file = path + '\\' + TrainFileNames[1]
        else:
            images_file = path + '\\' + TestFileNames[0]
            labels_file = path + '\\' + TestFileNames[1]

        images = np.fromfile(images_file, dtype=np.uint8)
        labels = np.fromfile(labels_file, dtype=np.uint8)

        self.images = torch.from_numpy(images).float().reshape(-1, 1, 28, 28)/255.
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform                  # transform modify the image
        self.target_transform = target_transform    # target_transform modify the label
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        kernel_sizes,
        paddings, 
        hidden_dims,
        num_classes
    ):
        super().__init__()

        layers = []
        current_channels = in_channels

        # Build conv stack dynamically
        for out_channels, k, p in zip(conv_channels, kernel_sizes, paddings):
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=k, padding=p))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            current_channels = out_channels

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()

        # Linear layer
        self.linearDim = None
        self._infer_flatten_size()
        linearLayers = []

        for dim in hidden_dims:
            linearLayers.append(nn.Linear(self.linearDim, dim)),
            linearLayers.append(nn.ReLU())
            self.linearDim = dim
        linearLayers.append(nn.Linear(self.linearDim, num_classes))

        self.classifier = nn.Sequential(*linearLayers)

    def _infer_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.features(x)
            self.linearDim = x.numel()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)
    
    def training_process(self, data, batch_size, learning_rate, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), learning_rate)

        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        datasize = len(data)

        loss_history = []
        accuracy_history = []

        for epoch in trange(epochs, desc="Training progress"):
            running_loss = 0
            correct = 0
            
            self.train()

            for X, y in loader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                pred = self(X)
                loss = criterion(pred, y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X.size(0)
                correct += (pred.argmax(1) == y).sum().item()

            loss_history.append(running_loss / datasize)
            accuracy_history.append(correct / datasize)

        return loss_history, accuracy_history

    def testing_process(self, data, batch_size=64):
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        datasize = len(data)

        self.eval()

        running_loss = 0
        correct = 0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)

                pred = self(X)
                loss = criterion(pred, y)

                running_loss += loss.item() * X.size(0)
                correct += (pred.argmax(1) == y).sum().item()

        avg_loss = running_loss / datasize
        accuracy = correct / datasize

        return avg_loss, accuracy
    
    def save_weight(self, path):
        torch.save(self.state_dict(), path)

    def load_weight(self, path, map_location=device):
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.to(map_location)

    def clear_weight(self):
        def reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.apply(reset)