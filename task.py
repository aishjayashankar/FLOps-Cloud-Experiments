"""flops-infra-drift: A Flower / PyTorch app."""

import os
# Increase timeout for dataset download (20 minutes)
os.environ["HF_DATASETS_CONNECTION_TIMEOUT"] = "1200"
os.environ["HF_DATASETS_READ_TIMEOUT"] = "1200"


from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor



# Fixed vocabulary size for hashing trick
VOCAB_SIZE = 10000
MAX_LEN = 32

class TextClassifier(nn.Module):
    """Simple LSTM model for text classification."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=64, num_classes=14):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)
        # x: (batch_size, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (1, batch_size, hidden_dim)
        out = self.fc(h_n[-1])
        return out

# Alias Net to TextClassifier to avoid breaking updates in client_app.py
Net = TextClassifier

def tokenize_and_pad(text):
    """Simple tokenizer with hashing trick and padding."""
    tokens = text.lower().split()
    # Hashing trick: hash string to integer index
    token_ids = [hash(w) % VOCAB_SIZE for w in tokens]
    # Truncate
    token_ids = token_ids[:MAX_LEN]
    # Pad
    if len(token_ids) < MAX_LEN:
        token_ids += [0] * (MAX_LEN - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long)

fds = None  # Cache FederatedDataset


def prepare_dataset():
    """Download dataset if not already present."""
    # Local files used, no download needed
    print("Using local DBPedia dataset.")


def load_data(partition_id: int, num_partitions: int):
    """Load partition DBPedia data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, partition_by="label", alpha=0.1, seed=42
        )
        fds = FederatedDataset(
            dataset="csv",
            # Point to local files
            data_files={"train": "DBPedia/train.csv", "test": "DBPedia/test.csv"},
            partitioners={"train": partitioner},
        )
    
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    def apply_transforms(batch):
        """Tokenize text and map labels."""
        # Tokenize "content" column for DBPedia
        batch["text"] = [tokenize_and_pad(t) for t in batch["content"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # print(f"Partition {partition_id} has {len(partition_train_test['train'])} training samples and {len(partition_train_test['test'])} test samples.")
    
    # Custom collate_fn to stack tensors
    # Datasets with list of tensors need this to form batch tensors
    # But PyTorch DataLoader does this automatically if they are same size.
    # Since we pad to MAX_LEN, they are same size.
    
    trainloader = DataLoader(partition_train_test["train"], batch_size=128, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=128)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # Lower LR for LSTM
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            text = batch["text"] # already tensor due to transform
            labels = batch["label"]
            
            # If batch keys are lists of tensors (HuggingFace datasets default for custom objects), stack them
            # BUT: with_transform usually returns lists. DataLoader collates them. 
            # If tokenize_and_pad returns tensors, DataLoader stacks them.
            
            optimizer.zero_grad()
            loss = criterion(net(text.to(device)), labels.to(device))
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
            text = batch["text"].to(device)
            labels = batch["label"].to(device)
            outputs = net(text)
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
