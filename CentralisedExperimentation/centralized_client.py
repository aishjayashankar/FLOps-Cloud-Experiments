import time
import torch
import torchvision.models

from centralized_task import load_data, train, test
from collections import OrderedDict

class CentralizedClient():
    def __init__(self, net, trainloader, valloader):
        self.model = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("CentralizedClient::__init__() - Using device:", self.device)
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
        return (
            self.get_parameters(),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters):
        start_time = time.time()
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client took {runtime:.4f} seconds to evaluate.")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(client_number):
    # Load model and data
    net = torchvision.models.resnet18(num_classes=10)
    trainloader, valloader = load_data(client_number)

    # Return Client instance
    return CentralizedClient(net, trainloader, valloader)