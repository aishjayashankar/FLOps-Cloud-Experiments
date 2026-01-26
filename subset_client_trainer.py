import time
import torch

from collections import OrderedDict
from flops_infra_drift.task import train
from torch.utils.data import Dataset, DataLoader

class DictStyleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
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
        
        # Guard against empty trainloader
        if self.trainloader is None or len(self.trainloader.dataset) == 0:
            print("WARNING: SubsetClientTrainer has empty dataset. Returning garbage metrics.")
            return (parameters, 0, {"train_loss": 0.0})

        self.set_parameters(parameters)
        
        # Freeze Embedding layer for substitution training to avoid "Language Barrier" noise
        # Freeze Embedding layer for substitution training
        frozen_params = []
        for name, param in self.model.named_parameters():
            if "embedding" in name.lower():
                param.requires_grad = False
                frozen_params.append(name)
        
        # Check if trainloader has batches
        if len(self.trainloader) == 0:
             print("WARNING: SubsetClientTrainer trainloader has 0 batches.")
             return (self.get_parameters(), 0, {"train_loss": 0.0})

        try:
            train_loss = train(
                self.model,
                self.trainloader,
                1,
                self.device,
            )
        except ZeroDivisionError:
             print("WARNING: ZeroDivisionError in train.")
             train_loss = 0.0
        except Exception as e:
             print(f"ERROR: Exception in train: {e}")
             train_loss = 0.0
        finally:
             # Unfreeze parameters to leave model in clean state
             for name, param in self.model.named_parameters():
                 if name in frozen_params:
                     param.requires_grad = True

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
    # Iterating the original loader to find matching records
    for batch in trainloader:
        texts = batch["text"]
        labels = batch["label"]

        for text, label in zip(texts, labels):
            label_int = int(label.item())
            if label_int in subset_distribution and current_counts[label_int] < subset_distribution[label_int]:
                collected_inputs.append(text)
                collected_labels.append(label)
                current_counts[label_int] += 1
        
        # Optimization: Early break if satisfied
        if all(current_counts[l] >= subset_distribution[l] for l in subset_distribution):
            break 

    if not collected_inputs:
        print(f"CRITICAL: Failed to collect any samples for subset distribution: {subset_distribution}")
        # Return a DataLoader with empty dataset
        return DataLoader(DictStyleDataset([], []), batch_size=trainloader.batch_size)

    inputs_tensor = torch.stack(collected_inputs)
    labels_tensor = torch.stack(collected_labels)
    target_dataset = DictStyleDataset(inputs_tensor, labels_tensor)
    
    # Use smaller batch size if dataset is small?
    batch_size = min(32, len(target_dataset))
    if batch_size == 0: batch_size = 1
    
    targetDL = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    return targetDL

def get_subset_client_trainer(
        net,
        subset_distribution: dict,
        trainloader: DataLoader) -> SubsetClientTrainer:
    """
    Create a SubsetClientTrainer instance with the provided subset distribution and trainloader.
    """
    # print(f"Creating SubsetClientTrainer with subset distribution: {subset_distribution}")
    subset_trainloader = load_subset_data(subset_distribution, trainloader)
    return SubsetClientTrainer(net, subset_trainloader)