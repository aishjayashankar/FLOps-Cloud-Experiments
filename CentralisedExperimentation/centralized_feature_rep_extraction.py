import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define custom dataset loader for multi-channel transformed data
class LargeFileDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths) * 25000  # Approximate count per file

    def __getitem__(self, idx):
        file_idx = idx // 25000  # Determine which file
        image_idx = idx % 25000  # Determine image within file

        # Load file lazily
        data = np.load(self.file_paths[file_idx])["images"]  # Assuming stored as NumPy arrays
        image = data[image_idx]  # Shape: (H, W, 3)

        # Convert to PyTorch tensor & normalize per channel
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        if self.transform:
            image = self.transform(image)

        return image

# File paths for large dataset B (replace with actual paths)
file_paths_B = ["data/B_part1.npz", "data/B_part2.npz", "data/B_part3.npz", "data/B_part4.npz"]

# CIFAR-10 standard transformations
transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
dataset_A = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset_B = LargeFileDataset(file_paths_B, transform=transform)

# Define feature extractor (ResNet-18 backbone, modified for multi-channel)
feature_extractor = torchvision.models.resnet18(pretrained=True)
feature_extractor.fc = torch.nn.Identity()  # Remove classification layer
feature_extractor = feature_extractor.to(device).eval()

def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    features = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            feats = feature_extractor(images).detach().cpu()
            features.append(feats)

    return torch.cat(features).mean(dim=0)  # Compute mean feature vector

# Compute feature means on A and B
mean_A = extract_features(dataset_A)
mean_B = extract_features(dataset_B)

# Compute similarity (MMD with Gaussian Kernel)
def gaussian_kernel(x, y, sigma=1.0):
    return torch.exp(-torch.norm(x - y, p=2)**2 / (2 * sigma**2))

def compute_mmd(sample_A, sample_B):
    return gaussian_kernel(sample_A, sample_B).mean()

# Select subset B' with closest similarity to A
selected_indices_B_prime = []
for idx in range(len(dataset_B)):
    sample_B = dataset_B[idx].unsqueeze(0).to(device)  # Load single sample
    if compute_mmd(sample_B, mean_A) > compute_mmd(mean_B, mean_A):  # Keep closer samples
        selected_indices_B_prime.append(idx)

# Create new dataset B' from selected indices
dataset_B_prime = torch.utils.data.Subset(dataset_B, selected_indices_B_prime)

print(f"Selected subset size: {len(dataset_B_prime)} out of