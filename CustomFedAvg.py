# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Optional, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import aggregate
from flwr.server.strategy.aggregate import aggregate_inplace, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class CustomFedAvg(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.last_client_weights = {}
        from collections import defaultdict
        self.similarity_avg = defaultdict(lambda: defaultdict(float))
        self.similarity_count = defaultdict(lambda: defaultdict(int))

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config["current_round"] = server_round
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Track sampled clients for failure detection
        self.current_round_sampled_cids = [client.cid for client in clients]

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        config["current_round"] = server_round
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Filter out failures and get successful results
        successful_results = results
        
        # Step 1: Update History for successful clients
        import numpy as np
        
        def flatten_weights(weights):
            """Flatten a list of numpy arrays into a single 1D array."""
            return np.concatenate([w.flatten() for w in weights])

        def cosine_similarity(w1, w2):
            """Calculate cosine similarity between two weight vectors."""
            norm_w1 = np.linalg.norm(w1)
            norm_w2 = np.linalg.norm(w2)
            if norm_w1 == 0 or norm_w2 == 0:
                return 0.0
            return np.dot(w1, w2) / (norm_w1 * norm_w2)

        # Filter out failures and get successful results
        successful_results = results
        
        from collections import defaultdict
        
        successful_cids = set()
        current_round_weights = {}
        
        # Store weights for successful clients and prepare for pairwise calculation
        for client_proxy, fit_res in successful_results:
            cid = client_proxy.cid
            successful_cids.add(cid)
            current_weights = parameters_to_ndarrays(fit_res.parameters)
            flat_weights = flatten_weights(current_weights)
            current_round_weights[cid] = flat_weights
            
            # Update last known weights (still useful for fallback or other logic)
            self.last_client_weights[cid] = flat_weights
            log(WARNING, f"Success Client CID: {cid}")

        # Update pairwise similarity averages using the running average formula
        successful_clients_list = list(current_round_weights.keys())
        for i in range(len(successful_clients_list)):
            cid_i = successful_clients_list[i]
            weights_i = current_round_weights[cid_i]
            
            for j in range(i + 1, len(successful_clients_list)):
                cid_j = successful_clients_list[j]
                weights_j = current_round_weights[cid_j]
                
                # Instantaneous similarity s_{i,j}^t
                s_t = cosine_similarity(weights_i, weights_j)
                
                # Previous count N_{i,j}^{t-1}
                N_prev = self.similarity_count[cid_i][cid_j]
                
                # Previous average R_{i,j}^{t-1}
                R_prev = self.similarity_avg[cid_i][cid_j]
                
                # Calculate new average R_{i,j}^t
                # Formula: R^t = (N / (N + 1)) * R^{t-1} + (1 / (N + 1)) * s^t
                R_new = (N_prev / (N_prev + 1)) * R_prev + (1 / (N_prev + 1)) * s_t
                
                # Update symmetric similarity stats
                self.similarity_avg[cid_i][cid_j] = R_new
                self.similarity_count[cid_i][cid_j] += 1
                
                self.similarity_avg[cid_j][cid_i] = R_new
                self.similarity_count[cid_j][cid_i] += 1

        # Check if we have any failures that need substitution
        if failures and self.accept_failures:
            log(WARNING, f"Found {len(failures)} failures. Attempting substitution...")
            
            # Deduce failed clients
            failed_cids = [cid for cid in self.current_round_sampled_cids if cid not in successful_cids]
            
            if not successful_results:
                log(WARNING, "No successful results to substitute from. Skipping substitution.")
            else:
                import random
                
                # For each deduced failure, pick the best friend from successful results based on Average Similarity
                for failed_cid in failed_cids:
                    best_friend = None
                    best_avg_similarity = -1.0
                    
                    log(WARNING, f"--- Finding Best Friend for Failed Client {failed_cid} ---")
                    
                    # Iterate over all potential friends (currently successful clients)
                    for friend_proxy, friend_res in successful_results:
                        friend_cid = friend_proxy.cid
                        
                        # Retrieve the running average similarity
                        avg_similarity = self.similarity_avg[failed_cid][friend_cid]
                        
                        log(WARNING, f"Candidate Friend {friend_cid}: Running Avg Similarity = {avg_similarity:.4f}")
                        
                        if avg_similarity > best_avg_similarity:
                            best_avg_similarity = avg_similarity
                            best_friend = (friend_proxy, friend_res)
                    
                    if best_friend and best_avg_similarity > -1.0: # Ensure we found at least one match with history
                        friend_client_proxy, friend_fit_res = best_friend
                        log(WARNING, f">>> Substituting failed client {failed_cid} with BEST friend {friend_client_proxy.cid} (Running Avg Similarity: {best_avg_similarity:.4f})")
                    else:
                        # Fallback to random if no history or no match found
                        friend_client_proxy, friend_fit_res = random.choice(successful_results)
                        log(WARNING, f">>> Substituting failed client {failed_cid} with RANDOM friend {friend_client_proxy.cid} (No history available)")

                    # Create a substitute result
                    successful_results.append((friend_client_proxy, friend_fit_res))

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(successful_results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in successful_results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in successful_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
