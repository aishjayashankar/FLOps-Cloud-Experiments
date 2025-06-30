import time
import torch
import torchvision.models
import pandas as pd
import numpy as np

from flops_infra_drift.task import train, test
from collections import OrderedDict
from torch.utils.data import DataLoader

class ClientSubsetTrainer():
    def __init__(self, net, trainloader):
        self.model = net
        self.trainloader = trainloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters):
        start_time = time.time()
        self.set_parameters(parameters)
        train_loss = train(
            self.model,
            self.trainloader,
            1,
            self.device,
        )
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client took {runtime:.4f} seconds to fit.")
        return self.get_parameters()

def format_data(df):
    print("Formatting data with shape:", df.shape)
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
        print("Dropped 'index' column.")

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
    print("Stacked image tensor shape:", X_tensor.shape)

    # Assume last column is label
    y = df.iloc[:, -1].values
    y_tensor = torch.tensor(y, dtype=torch.long)
    print("Labels tensor shape:", y_tensor.shape)

    # Return a dataset that yields dicts with keys "img" and "label"
    class DictTensorDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return {"image": self.X[idx], "label": self.y[idx]}

    return DictTensorDataset(X_tensor, y_tensor)

def load_data():
    train_file_name = "subset.csv"
    train_df = pd.read_csv(train_file_name)

    print(f"Loading training data from {train_file_name}")
    trainloader = DataLoader(format_data(train_df), batch_size=32, shuffle=True)
    return trainloader

def client_fn():
    # Load model and data
    net = torchvision.models.mobilenet_v3_small(num_classes=10)
    trainloader = load_data()

    # Return Client instance
    return ClientSubsetTrainer(net, trainloader)