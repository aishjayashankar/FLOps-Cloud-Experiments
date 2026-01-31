
from typing import Union, Optional
import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.strategy import Strategy


class MimiC(Strategy):
    """MimiC: Combating Client Dropouts in Federated Learning by Mimicking Central Updates.
    
    See Algorithm 1 in References/MimiC.pdf
    """
    def __init__(self, *, aggregator_strategy: Strategy = FedAvg) -> None:
        super().__init__()
        self.aggregator_strategy = aggregator_strategy
        self.global_parameters = None
        # Correction variables: dictionary mapping cid to list of numpy arrays
        self.correction_variables = {} 

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        self.global_parameters = self.aggregator_strategy.initialize_parameters(client_manager)
        return self.global_parameters

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[tuple[float, dict[str, Scalar]]]:
        return self.aggregator_strategy.evaluate(server_round, parameters)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> list[tuple[ClientProxy, FitIns]]:
        self.global_parameters = parameters
        return self.aggregator_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> list[tuple[ClientProxy, EvaluateIns]]:
        return self.aggregator_strategy.configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_evaluate(self, server_round: int, results: list[tuple[ClientProxy, EvaluateRes]], failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]]) -> tuple[Optional[float], dict[str, Scalar]]:
        return self.aggregator_strategy.aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[Union[tuple[ClientProxy, FitRes], BaseException]]) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using MimiC algorithm."""
        
        if not results:
            return None, {}

        # 1. Convert current global model to ndarrays
        current_weights = parameters_to_ndarrays(self.global_parameters)
        
        # 2. Process results from active clients
        # v_t^i = g_i + c_i
        # g_i = w_t - w_{t+1}^i
        
        modified_updates_v_t_i = []
        active_cids = []
        
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            active_cids.append(cid)
            
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            
            # g_i = current_weights - client_weights
            raw_update_g_i = [w_global - w_client for w_global, w_client in zip(current_weights, client_weights)]
            
            # Get correction variable c_i, init to zeros if not present
            if cid not in self.correction_variables:
                self.correction_variables[cid] = [np.zeros_like(w) for w in current_weights]
            
            c_i = self.correction_variables[cid]
            
            # v_t^i = g_i + c_i
            modified_update_v_t_i = [g + c for g, c in zip(raw_update_g_i, c_i)]
            modified_updates_v_t_i.append(modified_update_v_t_i)
            
        # 3. Compute global update v_t = Average(v_t^i)
        # Using simple average as per Algorithm 1 (Mean of modified updates)
        num_active = len(modified_updates_v_t_i)
        
        # Initialize v_t with zeros
        global_update_v_t = [np.zeros_like(w) for w in current_weights]
        
        for update in modified_updates_v_t_i:
            for idx, layer in enumerate(update):
                global_update_v_t[idx] += layer
                
        # Divide by number of active clients
        global_update_v_t = [layer / num_active for layer in global_update_v_t]
        
        # 4. Update Global Model w_{t+1} = w_t - v_t
        new_weights = [w_global - v_layer for w_global, v_layer in zip(current_weights, global_update_v_t)]
        
        # 5. Update correction variables
        # c_i <- v_t - g_i
        # g_i = v_t^i - c_i_old
        # c_i_new = v_t - (v_t^i - c_i_old) = v_t - v_t^i + c_i_old
        
        for i, cid in enumerate(active_cids):
            v_t_i = modified_updates_v_t_i[i]
            c_i_old = self.correction_variables[cid]
            
            c_i_new = [v - v_i + c_old for v, v_i, c_old in zip(global_update_v_t, v_t_i, c_i_old)]
            self.correction_variables[cid] = c_i_new
            
        # 6. Return new global parameters
        new_parameters = ndarrays_to_parameters(new_weights)
        
        # Aggregate metrics
        metrics_aggregated = {}
        if hasattr(self.aggregator_strategy, "fit_metrics_aggregation_fn") and self.aggregator_strategy.fit_metrics_aggregation_fn:
             fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
             metrics_aggregated = self.aggregator_strategy.fit_metrics_aggregation_fn(fit_metrics)

        print(f"MimiC Round {server_round}: Aggregated {len(results)} clients.")
        
        return new_parameters, metrics_aggregated
