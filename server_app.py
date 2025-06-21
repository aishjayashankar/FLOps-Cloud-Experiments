"""flops-infra-drift: A Flower / PyTorch app."""


from CustomFedAvgM import CustomFedAvgM
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from typing import List, Tuple
from flwr.common import Metrics
import torch
import torchvision.models
from flwr.common import ndarrays_to_parameters

def weighted_average_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)} if examples else {}

def weighted_average_train_loss(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics if "train_loss" in m]
    examples = [num_examples for num_examples, m in metrics if "train_loss" in m]
    return {"train_loss": sum(losses) / sum(examples)} if examples else {}

def get_initial_parameters():
    # Use the same model as the clients
    model = torchvision.models.resnet18(num_classes=10)
    # Convert model parameters to a list of numpy arrays
    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(params)

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Define strategy
    strategy = CustomFedAvgM(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=weighted_average_accuracy,
        fit_metrics_aggregation_fn=weighted_average_train_loss,
        initial_parameters=get_initial_parameters(),
    )
  
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)