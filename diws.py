import flops_infra_drift.consts as consts
import pickle

from concurrent.futures import ThreadPoolExecutor
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import  FedAvg
from flwr.server.strategy.aggregate import aggregate_inplace
from flwr.server.strategy.strategy import Strategy
from math import floor
from typing import Union
from typing import Optional, Union

# pylint: disable=line-too-long
class DIWS(Strategy):
    """Distribution Informed Weight Substitution (DIWS) strategy.

    Wrapper for any Flower Strategy that substitutes weights for dropped clients
    Parameters
    ----------
    aggregator_strategy : Strategy
        The strategy to be wrapped. This strategy will be used for the actual
        training and evaluation of the clients. The DIWS strategy will only
        substitute weights for dropped clients during the aggregation phase.
    global_parameters : Internally managed
        To be re-used during subset training for weight substitution
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        aggregator_strategy: Strategy = FedAvg
    ) -> None:
        super().__init__()
        self.aggregator_strategy = aggregator_strategy
        self.global_parameters = None
        self.label_distribution = {}

    def __repr__(self) -> str:
        return repr(self.aggregator_strategy)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        self.global_parameters = self.aggregator_strategy.initialize_parameters(client_manager)
        return self.global_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        return self.aggregator_strategy.evaluate(server_round, parameters)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        self.global_parameters = parameters
        return self.aggregator_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return self.aggregator_strategy.configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results from clients, substituting dropped clients if necessary."""

        # Initialize label distribution for the first round
        if server_round == 1:
            for client_proxy, fitres in results:
                client_label_distribution = pickle.loads(fitres.metrics.get("label_distribution"))
                self.label_distribution[client_proxy.cid] = client_label_distribution

        print(f"Number of results before substitution: {len(results)}")
        self.substitute_dropped_clients(server_round, results, failures)
        print(f"Number of results after substitution: {len(results)}")

        return self.aggregator_strategy.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        results = self.aggregator_strategy.aggregate_evaluate(server_round, results, failures)
        print(f"Results of aggregate evaluate: {results}")
        return results

    def substitute_dropped_clients(self,
                                server_round: int,
                                results: list[tuple[ClientProxy, FitRes]],
                                failures: list[Union[tuple[ClientProxy, FitRes], BaseException]]) -> None:
        """Substitute dropped clients with subset training, if required"""
        if (len(failures) == 0):
            print(f"No dropped clients to substitute in round {server_round}.")
            return
        
        active_clients_ids = [x[0].cid for x in results]
        dropped_clients_ids = set(self.label_distribution.keys()) - set(active_clients_ids)
        print(f"Dropped clients IDs: {dropped_clients_ids}")

        # Get active clients distribution
        _, active_clients_distribution = self.consolidate_label_distributions(active_clients_ids)

        for dropped_cid in dropped_clients_ids:
            print(f"Substituting for dropped client: {dropped_cid}")
            dropped_client_distribution = self.label_distribution.get(dropped_cid, {})
            
            client_subset_distributions = self.get_subset_distribution_for_active_clients(
                dropped_client_distribution, active_clients_distribution, active_clients_ids)

            outputs = []
            for client_proxy, _ in results:
                if client_proxy is None:
                    continue
                    
                # Check if this client needs to train (has non-empty distribution)
                if not client_subset_distributions.get(client_proxy.cid):
                    continue

                subset_distribution_bytes = pickle.dumps(client_subset_distributions[client_proxy.cid])
                config = {"subset_distribution": subset_distribution_bytes,
                        "custom_rpc": "handle_missing_clients"}
                fitIns = FitIns(parameters=self.global_parameters, config=config)
                
                # Sequential execution
                fitRes = client_proxy.fit(fitIns, consts.SUBSTITUTION_TIMEOUT, server_round)
                outputs.append(fitRes)
            
            if outputs:
                substituted_parameters_fitRes = self.aggregate_substitution_parameters(outputs)
                results.append((None, substituted_parameters_fitRes))
            else:
                print(f"No active clients could substitute for {dropped_cid}")


    def consolidate_label_distributions(self, active_clients_ids):
        dropped_clients_ids = set(self.label_distribution.keys()) - set(active_clients_ids)
        print(f"Dropped clients IDs: {dropped_clients_ids}")
        print(f"Active clients IDs: {active_clients_ids}")

        # Consolidate label distribution for dropped clients
        dropped_clients_distribution = {}
        for cid in dropped_clients_ids:
            client_dist = self.label_distribution.get(cid, {})
            for label, count in client_dist.items():
                dropped_clients_distribution[label] = dropped_clients_distribution.get(label, 0) + count

        # Consolidate label distribution for active clients
        active_clients_distribution = {}
        for cid in active_clients_ids:
            client_dist = self.label_distribution.get(cid, {})
            for label, count in client_dist.items():
                active_clients_distribution[label] = active_clients_distribution.get(label, 0) + count

        print(f"Dropped clients distribution: {dropped_clients_distribution}")
        print(f"Active clients distribution: {active_clients_distribution}")

        return dropped_clients_distribution, active_clients_distribution
    
    
    def get_consolidated_representative_distribution(self, dropped_clients_distribution, active_clients_distribution):
        # Calculate the representative subset distribution
        representative_subset_distribution = {}
        total_dropped = sum(dropped_clients_distribution.values())

        target_percentages = {
            label: count / total_dropped
            for label, count in dropped_clients_distribution.items()
        }

        anchor_label = max(
            dropped_clients_distribution,
            key=lambda label: dropped_clients_distribution[label],
        )

        representative_subset_distribution[anchor_label] = min(
            active_clients_distribution[anchor_label],
            dropped_clients_distribution[anchor_label])
        
        # Calculate the total count based on dropped percentage and chosen anchor label value
        anchor_label_total = floor(representative_subset_distribution[anchor_label] / target_percentages[anchor_label])

        for label, count in active_clients_distribution.items():
            if label == anchor_label:
                continue
            if label not in target_percentages:
                representative_subset_distribution[label] = 0
                continue
            target_count = floor(target_percentages[label] * anchor_label_total)
            representative_subset_distribution[label] = min(target_count, count)        

        print(f"Subset distribution for active clients: {representative_subset_distribution}")
        return representative_subset_distribution
    
    
    def get_subset_distribution_for_active_clients(
            self,
            dropped_clients_distribution: dict,
            active_clients_distribution: dict,
            active_clients_ids: list[str]) -> dict:
        """Get representative subset distribution for active clients."""
        
        representative_subset_distribution = self.get_consolidated_representative_distribution(
            dropped_clients_distribution, active_clients_distribution)        

        # Distribute the representative subset among active clients
        subset_distribution_per_client = {cid: {} for cid in active_clients_ids}
        for label, total_needed in representative_subset_distribution.items():
            client_counts = [
                (cid, self.label_distribution.get(cid, {}).get(label, 0))
                for cid in active_clients_ids
            ]
            idx = 0
            while total_needed > 0 and any(count > 0 for _, count in client_counts):
                cid, available = client_counts[idx % len(client_counts)]
                if available > 0:
                    subset_distribution_per_client[cid][label] = subset_distribution_per_client[cid].get(label, 0) + 1
                    client_counts[idx % len(client_counts)] = (cid, available - 1)
                    total_needed -= 1
                idx += 1

        print(f"Subset distribution per client: {subset_distribution_per_client}")
        return subset_distribution_per_client
    
    
    def aggregate_substitution_parameters(self, results: list[FitRes]) -> FitRes:
        results = [(None, fitRes) for fitRes in results]
        aggregated_parameters = aggregate_inplace(results)
        total_samples = sum(fitRes.num_examples for _, fitRes in results)

        aggregated_results = FitRes(
            parameters=ndarrays_to_parameters(aggregated_parameters),
            num_examples=total_samples,
            metrics=None,
            status=None
        )
        return aggregated_results