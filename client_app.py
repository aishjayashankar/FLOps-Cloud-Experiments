"""flops-infra-drift: A Flower / PyTorch app."""

import time
import torch
import torchvision.models
import pickle
import flops_infra_drift.consts as consts

from collections import OrderedDict
from collections import Counter
from flops_infra_drift.subset_client_trainer import get_subset_client_trainer
from flops_infra_drift.task import load_data, test, train, Net
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

def ShouldNodeDisconnect(partition_id, current_round, trainloader_size):
    start_disconnect = consts.CLIENT_DROP_ROUND_START
    end_disconnect = consts.CLIENT_DROP_ROUND_END
    return partition_id in consts.DROPPED_CLIENT_PARITIONS_IDS and (start_disconnect <= current_round < end_disconnect)

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        self.model = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.partition_id = partition_id

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

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Fit model parameters using local training data."""

        # Trigger subset trainer for missing clients
        if config.get("custom_rpc") == "handle_missing_clients":
            print("Perform fit on representative subset")
            subsetClientTrainer = get_subset_client_trainer(
                self.model,
                pickle.loads(config["subset_distribution"]),
                self.trainloader)
            # print(f"Calling subsetClientTrainer.fit with {type(parameters)}...")
            result = subsetClientTrainer.fit(parameters)
            # print(f"subsetClientTrainer.fit returned: {type(result)}")
            
            return result            
        
        start_time = time.time()

        # Simulating client disconnection
        if (ShouldNodeDisconnect(self.partition_id, config["current_round"], len(self.trainloader.dataset))):
            print("Disconnecting partition: ", self.partition_id, " for round: ", config["current_round"])
            return "Garbage"
        
        self.set_parameters(parameters)
        train_loss = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client: {self.partition_id} took {runtime:.4f} seconds to fit.")

        metrics = {"train_loss": train_loss}
        # Share label distribution if it's the first round
        if config["current_round"] == 1:
            label_distribution = self.get_label_distribution()
            metrics["label_distribution"] = pickle.dumps(label_distribution)

        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        start_time = time.time()
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        end_time = time.time()
        runtime = end_time - start_time
        # print(f"Client: {self.partition_id} took {runtime:.4f} seconds to evaluate.")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    def get_label_distribution(self) -> dict:
        """Calculate and return label distribution in the training data."""
        print("Calculating label distribution for partition:", self.partition_id)

        label_counter = Counter()
        for batch in self.trainloader:
            labels = batch["label"]
            label_counter.update([int(label) for label in labels])

        print(f"Label distribution for partition {self.partition_id}: {label_counter}")
        return dict(label_counter)


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(
        net, trainloader, valloader, local_epochs, partition_id
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
