"""flops-infra-drift: A Flower / PyTorch app."""

import time
import torch
import torchvision.models
import pickle
import flops_infra_drift.consts as consts
import flops_infra_drift.client_subset_trainer as cst

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flops_infra_drift.task import Net, get_weights, load_data, set_weights, test, train
from collections import OrderedDict


def ShouldNodeDisconnect(partition_id, current_round):
    if partition_id != 3:
        return False
    # For node n, partition_id is n-1
    # start_disconnect = 5, 6, 7 for partition_ids 2, 3, 4
    # start_disconnect = 7 #(partition_id + 3)
    # end_disconnect = 31

    return (
        consts.CLIENT_DROP_ROUND_START <= current_round < consts.CLIENT_DROP_ROUND_END
    )


def get_label_distribution(trainloader):
    print("Calculating label distribution...")
    label_counts = {}
    total = 0
    for batch in trainloader:
        labels = batch["label"]
        for label in labels:
            label_int = int(label)
            label_counts[label_int] = label_counts.get(label_int, 0) + 1
            total += 1
    print(f"Total samples counted: {total}")
    distribution = {}
    for label, count in label_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        distribution[label] = (count, percentage)
        print(f"Label {label}: count = {count}, percentage = {percentage:.2f}%")
    print("Label distribution calculation complete.")
    return distribution


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
        start_time = time.time()
        # Simulating client disconnection
        if ShouldNodeDisconnect(self.partition_id, config["current_round"]):
            print(
                "Disconnecting partition: ",
                self.partition_id,
                " for round: ",
                config["current_round"],
            )
            return "Garbage"
        self.set_parameters(parameters)
        parameters_copy = self.get_parameters({})
        train_loss = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        # Train and get parameters for dropped client
        dropped_client_parameters = None
        if (
            consts.CLIENT_DROP_ROUND_START
            <= config["current_round"]
            < consts.CLIENT_DROP_ROUND_END
        ) and (self.partition_id == consts.DROPPED_CLIENT_SUBSTITUTE_PARTITION_ID):
            print(
                f"Training dropped client parameters for round: {config['current_round']} in partition: {self.partition_id}"
            )
            client_subset_trainer = cst.client_fn()
            dropped_client_parameters = client_subset_trainer.fit(parameters_copy)

        # Serialize dropped client parameters
        dropped_client_parameters_bytes = None
        if dropped_client_parameters:
            print(
                f"Serializing dropped client parameters for round: {config['current_round']} in partition: {self.partition_id}"
            )
            dropped_client_parameters_bytes = pickle.dumps(dropped_client_parameters)

        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client: {self.partition_id} took {runtime:.4f} seconds to fit.")

        metrics = {"train_loss": train_loss}
        if dropped_client_parameters_bytes is not None:
            metrics["dropped_client_parameters_bytes"] = dropped_client_parameters_bytes
        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        start_time = time.time()
        # Simulating client disconnection
        # if (ShouldNodeDisconnect(self.partition_id, config["current_round"])):
        #     print("Disconnecting partition: ", self.partition_id, " for round: ", config["current_round"])
        #     return "Garbage"
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client: {self.partition_id} took {runtime:.4f} seconds to evaluate.")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = torchvision.models.resnet18(num_classes=10)
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
