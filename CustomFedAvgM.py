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
"""Federated Averaging with Server Momentum (FedAvgM) strategy.

Implementation based on FedAvg with server-side momentum optimization.
"""


from logging import WARNING
from typing import Callable, Optional, Union

import numpy as np

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
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class CustomFedAvgM(Strategy):
    """Federated Averaging with Server Momentum (FedAvgM) strategy.

    Implementation of FedAvg with server-side momentum for improved convergence.

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
    server_learning_rate : float, optional
        Server-side learning rate for applying momentum updates. Defaults to 1.0.
    server_momentum : float, optional
        Server-side momentum factor. Should be between 0 and 1. Defaults to 0.0.
        Setting to 0.0 reduces to standard FedAvg.
    """

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
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
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
        
        # FedAvgM specific parameters
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        
        # Server-side momentum state
        self.momentum_vector: Optional[NDArrays] = None
        self.current_weights: Optional[NDArrays] = None

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"CustomFedAvgM(accept_failures={self.accept_failures}, server_momentum={self.server_momentum})"
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
        if initial_parameters is not None:
            # Store the initial weights for FedAvgM server-side momentum
            self.current_weights = parameters_to_ndarrays(initial_parameters)
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
        """Aggregate fit results using weighted average and server-side momentum."""
        # Handle case where no successful results
        if not results:
            log(WARNING, f"Round {server_round}: No successful client results to aggregate")
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            log(
                WARNING,
                f"Round {server_round}: {len(failures)} clients failed and "
                f"accept_failures=False, skipping aggregation",
            )
            return None, {}

        # Log failures if any occurred but we're accepting them
        if failures:
            log(
                WARNING,
                f"Round {server_round}: {len(failures)} clients failed, "
                f"but continuing with {len(results)} successful clients. "
                f"Failed clients will be ignored in aggregation.",
            )

        # Get current server weights
        if self.current_weights is None:
            # If we don't have stored weights, use the first client's weights as baseline
            log(WARNING, "No stored server weights found, using first client's weights as baseline")
            self.current_weights = parameters_to_ndarrays(results[0][1].parameters)

        # Convert results to get weight updates (only from successful clients)
        weights_results = []
        for client_proxy, fit_res in results:
            try:
                client_weights = parameters_to_ndarrays(fit_res.parameters)
                weights_results.append((client_weights, fit_res.num_examples))
            except Exception as e:
                log(
                    WARNING,
                    f"Round {server_round}: Failed to extract parameters from client "
                    f"{client_proxy.cid}, skipping this client. Error: {e}",
                )
                continue
        
        # Check if we have any valid weights after extraction
        if not weights_results:
            log(
                WARNING,
                f"Round {server_round}: No valid client weights could be extracted, "
                f"cannot perform aggregation",
            )
            return None, {}

        # Aggregate client weights using weighted average
        aggregated_ndarrays = aggregate(weights_results)
        
        # Apply server-side momentum
        updated_weights = self._apply_server_momentum(aggregated_ndarrays)
        
        # Update stored server weights for next round
        self.current_weights = updated_weights
        
        parameters_aggregated = ndarrays_to_parameters(updated_weights)

        # Aggregate custom metrics if aggregation fn was provided (only from successful clients)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = []
            for client_proxy, fit_res in results:
                try:
                    if fit_res.num_examples > 0 and fit_res.metrics is not None:
                        fit_metrics.append((fit_res.num_examples, fit_res.metrics))
                except Exception as e:
                    log(
                        WARNING,
                        f"Round {server_round}: Failed to extract fit metrics from client "
                        f"{client_proxy.cid}, skipping this client. Error: {e}",
                    )
                    continue
            
            if fit_metrics:
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            else:
                log(
                    WARNING,
                    f"Round {server_round}: No valid fit metrics could be extracted",
                )
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
        # Handle case where no successful results
        if not results:
            log(WARNING, f"Round {server_round}: No successful evaluation results to aggregate")
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            log(
                WARNING,
                f"Round {server_round}: {len(failures)} evaluation clients failed and "
                f"accept_failures=False, skipping evaluation aggregation",
            )
            return None, {}

        # Log failures if any occurred but we're accepting them
        if failures:
            log(
                WARNING,
                f"Round {server_round} evaluation: {len(failures)} clients failed, "
                f"but continuing with {len(results)} successful clients. "
                f"Failed clients will be ignored in evaluation.",
            )

        # Aggregate loss (only from successful clients)
        loss_results = []
        for client_proxy, evaluate_res in results:
            try:
                if evaluate_res.loss is not None and evaluate_res.num_examples > 0:
                    loss_results.append((evaluate_res.num_examples, evaluate_res.loss))
                else:
                    log(
                        WARNING,
                        f"Round {server_round}: Invalid loss or num_examples from client "
                        f"{client_proxy.cid}, skipping this client",
                    )
            except Exception as e:
                log(
                    WARNING,
                    f"Round {server_round}: Failed to extract loss from client "
                    f"{client_proxy.cid}, skipping this client. Error: {e}",
                )
                continue
        
        # Check if we have any valid losses after extraction
        if not loss_results:
            log(
                WARNING,
                f"Round {server_round}: No valid evaluation losses could be extracted, "
                f"cannot perform evaluation aggregation",
            )
            return None, {}
            
        loss_aggregated = weighted_loss_avg(loss_results)

        # Aggregate custom metrics if aggregation fn was provided (only from successful clients)
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = []
            for client_proxy, evaluate_res in results:
                try:
                    if evaluate_res.num_examples > 0 and evaluate_res.metrics is not None:
                        eval_metrics.append((evaluate_res.num_examples, evaluate_res.metrics))
                except Exception as e:
                    log(
                        WARNING,
                        f"Round {server_round}: Failed to extract metrics from client "
                        f"{client_proxy.cid}, skipping this client. Error: {e}",
                    )
                    continue
            
            if eval_metrics:
                metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            else:
                log(
                    WARNING,
                    f"Round {server_round}: No valid evaluation metrics could be extracted",
                )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def _apply_server_momentum(self, aggregated_weights: NDArrays) -> NDArrays:
        """Apply server-side momentum to the aggregated weights.
        
        Implements server-side momentum as:
        momentum_vector = server_momentum * momentum_vector + server_learning_rate * delta
        new_weights = current_weights + momentum_vector
        
        where delta = aggregated_weights - current_weights
        """
        if self.server_momentum == 0.0:
            # No momentum, return weighted average of aggregated weights and current weights
            return [
                self.server_learning_rate * agg_w + (1 - self.server_learning_rate) * curr_w
                for agg_w, curr_w in zip(aggregated_weights, self.current_weights)
            ]
        
        # Compute the update direction (pseudo-gradient)
        delta = [
            agg_w - curr_w 
            for agg_w, curr_w in zip(aggregated_weights, self.current_weights)
        ]
        
        # Initialize momentum vector if not already done
        if self.momentum_vector is None:
            self.momentum_vector = [np.zeros_like(d) for d in delta]
        
        # Update momentum vector and apply to weights
        updated_weights = []
        new_momentum_vector = []
        
        for curr_w, d, m_prev in zip(self.current_weights, delta, self.momentum_vector):
            # Update momentum: momentum = β * momentum_prev + η * delta
            momentum = self.server_momentum * m_prev + self.server_learning_rate * d
            new_momentum_vector.append(momentum)
            
            # Update weights: w_new = w_current + momentum
            new_weight = curr_w + momentum
            updated_weights.append(new_weight)
        
        # Store updated momentum vector
        self.momentum_vector = new_momentum_vector
        
        return updated_weights