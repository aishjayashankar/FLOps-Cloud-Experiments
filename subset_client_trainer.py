import time
import torch

from collections import OrderedDict
from flops_infra_drift.task import train
from torch.utils.data import Dataset, DataLoader

class DictStyleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

class SubsetClientTrainer:
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
        print(f"Subset client trainer took {runtime:.4f} seconds to fit.")

        metrics = {"train_loss": train_loss}
        return (self.get_parameters(), len(self.trainloader.dataset), metrics)

def load_subset_data(
        subset_distribution: dict,
        trainloader: DataLoader) -> DataLoader:
    """
    This method is called when a client receives a custom RPC to handle missing clients.
    It trains the model using a representative subset of active clients.
    """

    current_counts = {label: 0 for label in subset_distribution}
    collected_inputs = []
    collected_labels = []

    # Collect required number of records
    for batch in trainloader:
        images = batch["image"]
        labels = batch["label"]

        for img, label in zip(images, labels):
            label_int = int(label.item())
            if label_int in subset_distribution and current_counts[label_int] < subset_distribution[label_int]:
                collected_inputs.append(img)
                collected_labels.append(label)
                current_counts[label_int] += 1

            if all(current_counts[l] >= subset_distribution[l] for l in subset_distribution):
                break
        if all(current_counts[l] >= subset_distribution[l] for l in subset_distribution):
            break 

    inputs_tensor = torch.stack(collected_inputs)
    labels_tensor = torch.stack(collected_labels)
    target_dataset = DictStyleDataset(inputs_tensor, labels_tensor)
    targetDL = DataLoader(target_dataset, batch_size=trainloader.batch_size, shuffle=False)

    return targetDL

def get_subset_client_trainer(
        net,
        subset_distribution: dict,
        trainloader: DataLoader) -> SubsetClientTrainer:
    """
    Create a SubsetClientTrainer instance with the provided subset distribution and trainloader.
    """
    print(f"Creating SubsetClientTrainer with subset distribution: {subset_distribution}")
    subset_trainloader = load_subset_data(subset_distribution, trainloader)
    return SubsetClientTrainer(net, subset_trainloader)