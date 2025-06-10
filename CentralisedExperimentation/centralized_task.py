import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging

from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.data import TensorDataset

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def format_data(df):
    logging.info("Formatting data with shape: %s", df.shape)
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
        logging.debug("Dropped 'index' column.")

    # Create a dict where each key is a channel name and value is a tensor of shape (num_samples, 32*32)
    img_cols = df.columns[:3]
    img_dict = {}
    # Parse each channel column into a tensor of shape (num_samples, 32, 32)
    channel_tensors = []
    for c in img_cols:
        # Each cell is a string representation of a 2D list, e.g., "[[1, 2, ...], ...]"
        channel_data = df[c].apply(lambda x: np.array(eval(x), dtype=np.float32))
        channel_tensor = torch.stack([torch.tensor(arr) for arr in channel_data.tolist()])
        channel_tensors.append(channel_tensor)
    # Stack channel tensors into a single tensor of shape (num_samples, 3, 32, 32)
    X_tensor = torch.stack(channel_tensors, dim=1)
    logging.info("Stacked image tensor shape: %s", X_tensor.shape)

    # Assume last column is label
    y = df.iloc[:, -1].values
    y_tensor = torch.tensor(y, dtype=torch.long)
    logging.info("Labels tensor shape: %s", y_tensor.shape)

    # Return a dataset that yields dicts with keys "img" and "label"
    class DictTensorDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return {"img": self.X[idx], "label": self.y[idx]}

    return DictTensorDataset(X_tensor, y_tensor)


def load_data(client_number):
    # train_folder_path = "/home/ketanatri/Desktop/PhD/SplitData/"
    # train_folder_path = "/mnt/Users/Ketan/Desktop/PhD/FLOpsInfraDrift/RunArtifacts/ConsolidatedData/TrainTestData/"
    train_folder_path = "./"
    train_file_name = "balanced_train_data.csv"
    #train_file_name = f"{client_number}_subset.csv"
    train_df = pd.read_csv(train_folder_path + train_file_name)
    folder_path = "/mnt/Users/Ketan/Desktop/PhD/FLOpsInfraDrift/RunArtifacts/ConsolidatedData/TrainTestData/"
    test_data_path = folder_path + "3-test-data.csv"
    test_df = pd.read_csv(test_data_path)

    logging.info(f"Loading training data from {train_file_name}")
    trainloader = DataLoader(format_data(train_df), batch_size=32, shuffle=True)
    logging.info("Loading test data from %s", test_data_path)
    testloader = DataLoader(format_data(test_df), batch_size=32, shuffle=False)
    logging.info("Created trainloader and testloader with batch size 32.")
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
