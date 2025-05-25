"""flops-infra-drift: A Flower / PyTorch app."""

import time
import torch
import torchvision.models

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flops_infra_drift.task import Net, get_weights, load_data, set_weights, test, train
from collections import OrderedDict
import csv


def ShouldNodeDisconnect(partition_id, current_round):
    return False
    if partition_id != 3:
        return False
    # For node n, partition_id is n-1
    # start_disconnect = 5, 6, 7 for partition_ids 2, 3, 4
    start_disconnect = 7  # (partition_id + 3)
    end_disconnect = 31

    return start_disconnect <= current_round < end_disconnect


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

        # Save parameters to CSV for the first round
        if config["current_round"] == 1:
            csv_filename = f"{self.partition_id}-parameters-round-0.csv"
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Parameter Name", "Values"])
                parameters = self.get_parameters({})
                for name, param in zip(self.model.state_dict().keys(), parameters):
                    writer.writerow([name, param.tolist()])

        train_loss = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client: {self.partition_id} took {runtime:.4f} seconds to fit.")

        # Save parameters to a separate CSV file for each round
        csv_filename = (
            f"{self.partition_id}-parameters-round-{config['current_round']}.csv"
        )
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Parameter Name", "Values"])
            parameters = self.get_parameters({})
            for name, param in zip(self.model.state_dict().keys(), parameters):
                writer.writerow([name, param.tolist()])

        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
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

        # Print sample values from parameters
        parameters = self.get_parameters({})
        # print(
        #     f"Sample values from parameters:\nparameters[0][1][2][3][4]: {parameters[0][1][2][3][4]}\nparameters:[1][2]: {parameters[1][2]}\nparameters[2][3]: {parameters[2][3]}\nparameters[3][4]: {parameters[3][4]}"
        # )
        # Save parameters to a separate CSV file for each round
        if self.partition_id == 1:
            csv_filename = f"aggregated-parameters-round-{config['current_round']}.csv"
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Parameter Name", "Values"])
                for name, param in zip(self.model.state_dict().keys(), parameters):
                    writer.writerow([name, param.tolist()])

        loss, accuracy = test(self.model, self.valloader, self.device)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client: {self.partition_id} took {runtime:.4f} seconds to evaluate.")

        # Save loss and accuracy to CSV
        csv_filename = f"{self.partition_id}-loss-accuracy.csv"
        file_exists = False
        try:
            with open(csv_filename, mode="r") as file:
                file_exists = True
        except FileNotFoundError:
            pass

        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Round", "Loss", "Dataset Size", "Accuracy"])
            writer.writerow(
                [config["current_round"], loss, len(self.valloader.dataset), accuracy]
            )

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
