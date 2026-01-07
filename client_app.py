"""flops-infra-drift: A Flower / PyTorch app."""

import time
import torch
import torchvision.models
import pickle
import flops_infra_drift.consts as consts
import copy

from collections import OrderedDict
from collections import Counter
from flops_infra_drift.subset_client_trainer import get_subset_client_trainer
from flops_infra_drift.task import load_data, test, train
from flwr.client import ClientApp, NumPyClient
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import flops_infra_drift.keys as keys
import tenseal as ts

def ShouldNodeDisconnect(partition_id, current_round):
    if partition_id not in consts.DROPPED_CLIENT_PARITIONS_IDS:
        return False

    return (
        consts.CLIENT_DROP_ROUND_START <= current_round < consts.CLIENT_DROP_ROUND_END
    )

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        self.model = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Client {partition_id} initialized on device: {self.device} (CUDA available: {torch.cuda.is_available()})")
        self.model.to(self.device)
        self.partition_id = partition_id
        
        # Load FHE Context (Secret key needed for decryption/encryption)
        self.context = keys.load_context(consts.CLIENT_CONTEXT_PATH)

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
            # Decrypt target needed
            encrypted_target = pickle.loads(config["subset_distribution"])
            target_distribution = {}
            for label, enc_data in encrypted_target.items():
                 # Decrypt: Result is a vector, take 0th element
                 val = ts.ckks_vector_from(self.context, enc_data).decrypt()[0]
                 target_distribution[label] = max(0, int(round(val)))

            print("Perform fit on representative subset", target_distribution)
            # Use deepcopy to prevent in-place modification of the client's main model
            subsetClientTrainer = get_subset_client_trainer(
                copy.deepcopy(self.model),
                target_distribution,
                self.trainloader)
            return subsetClientTrainer.fit(parameters)            
        
        start_time = time.time()

        # Simulating client disconnection
        if ShouldNodeDisconnect(self.partition_id, config["current_round"]):
            print(
                "Disconnecting partition: ",
                self.partition_id,
                " for round: ",
                config["current_round"],
            )
            # Return dropped signal instead of raising exception to preserve simulation flow
            return [], 0, {"is_dropped": True}
        
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
        
        # Explicit GC to prevent memory leaks in Ray actors
        import gc
        gc.collect()

        metrics = {"train_loss": train_loss}
        # Share label distribution if it's the first round
        if config["current_round"] == 1:
            label_distribution = self.get_label_distribution()
            
            # Encrypt label distribution
            encrypted_dist = {}
            for label, count in label_distribution.items():
                encrypted_dist[label] = ts.ckks_vector(self.context, [count]).serialize()
                
            metrics["label_distribution"] = pickle.dumps(encrypted_dist)
            metrics["partition_id"] = str(self.partition_id)

        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        # Masked Interactive Protocol Check
        if "blinded_diff" in config:
            blinded_diff_map = pickle.loads(config["blinded_diff"])
            is_capped_map = {}
            for label, enc_diff in blinded_diff_map.items():
                # Decrypt: (Fair - Stock) * Mask
                # Mask is positive, so sign is preserved.
                # If > 0: Capped (Fair > Stock)
                # If < 0: Capable (Stock > Fair)
                val = ts.ckks_vector_from(self.context, enc_diff).decrypt()[0]
                is_capped_map[label] = (val > 0)
            
            return float(0.0), 0, {"is_capped": pickle.dumps(is_capped_map)}

        # Distributed Target Scaling Check
        if "check_global_feasibility" in config:
            blinded_checks_map = pickle.loads(config["check_global_feasibility"])
            # Returns simple boolean: Is the proposal feasible?
            # Proposal is feasible if ALL checks are >= 0
            is_feasible = True
            for label, enc_val in blinded_checks_map.items():
                 # Decrypt: Active - (Dropped * k)
                 # Masked with positive random value
                 val = ts.ckks_vector_from(self.context, enc_val).decrypt()[0]
                 # Use epsilon for robustness against FHE noise
                 if val < -0.01: # Means Active < Dropped * k (significantly)
                     is_feasible = False
                     break
            
            return float(0.0), 0, {"is_feasible": is_feasible}

        start_time = time.time()
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Client: {self.partition_id} took {runtime:.4f} seconds to evaluate.")
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
