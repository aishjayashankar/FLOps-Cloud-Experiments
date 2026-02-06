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
import tenseal as ts
import pickle
import random
import os
import flops_infra_drift.consts as consts
import flops_infra_drift.keys as keys
import datetime

def log_debug(msg):
    with open("debug_diws.log", "a") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")

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
        self.context = None
        self.computation_cache = {}

    def __repr__(self) -> str:
        return repr(self.aggregator_strategy)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        self.global_parameters = self.aggregator_strategy.initialize_parameters(client_manager)
        # Load server context
        if not os.path.exists(consts.SERVER_CONTEXT_PATH):
            print("Generating new FHE keys...")
            keys.create_and_save_context(consts.SERVER_CONTEXT_PATH, consts.CLIENT_CONTEXT_PATH)

        if self.context is None:
             self.context = keys.load_context(consts.SERVER_CONTEXT_PATH)
        self.cid_to_partition = {}
        self.inv_num_cache = {}
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
        
        # Filter out clients that signaled dropout
        valid_results = []
        for client_proxy, fit_res in results:
             if fit_res.metrics.get("is_dropped"):
                 print(f"Detected dropout signal from {client_proxy.cid}")
             else:
                 valid_results.append((client_proxy, fit_res))
        
        results = valid_results

        # Initialize label distribution for the first round
        if server_round == 1:
            for client_proxy, fitres in results:
                # Deserialize encrypted distribution
                client_dist_bytes = pickle.loads(fitres.metrics.get("label_distribution"))
                client_dist = {}
                for label, enc_bytes in client_dist_bytes.items():
                    client_dist[label] = ts.ckks_vector_from(self.context, enc_bytes)
                
                # Use partition ID for storage if available
                pid = fitres.metrics.get("partition_id")
                if pid:
                    self.cid_to_partition[client_proxy.cid] = str(pid)
                    self.label_distribution[str(pid)] = client_dist
                    print(f"Mapped CID {client_proxy.cid} to Partition {pid}")
                else:
                    self.label_distribution[client_proxy.cid] = client_dist
                    print(f"Warning: No partition ID for {client_proxy.cid}")

        print(f"Number of results before substitution: {len(results)}")
        if server_round >= consts.CLIENT_DROP_ROUND_START and server_round < consts.CLIENT_DROP_ROUND_END:
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
        
        dropped_cids = [str(cid) for cid in consts.DROPPED_CLIENT_PARITIONS_IDS]
        active_client_proxies = [res[0] for res in results]
        active_cids = [p.cid for p in active_client_proxies]
        
        active_pids = []
        for cid in active_cids:
            if cid in self.cid_to_partition:
                active_pids.append(self.cid_to_partition[cid])
            else:
                active_pids.append(str(cid)) # Fallback
                
        actual_dropped = [pid for pid in dropped_cids if pid not in active_pids]
        
        if not actual_dropped:
             print("No dropped clients to substitute.")
             return

        # Generate Cache Key
        cache_key = hash((tuple(sorted(active_cids)), tuple(sorted(actual_dropped))))
        
        if cache_key in self.computation_cache:
            print("Cache hit! Using cached substitution shares.")
            log_debug("Cache hit! Using cached substitution shares.")
            final_shares = self.computation_cache[cache_key]
        else:
            print("Cache miss. Computing substitution shares...")
            final_shares = self._compute_substitution_shares(active_cids, active_client_proxies, actual_dropped, server_round)
            self.computation_cache[cache_key] = final_shares



        with ThreadPoolExecutor() as executor:
            futures = []
            for client_proxy in active_client_proxies:
                cid = client_proxy.cid
                share_map = final_shares.get(cid, {})
                
                # Serialize shares
                serialized_shares = {l: v.serialize() for l, v in share_map.items()}
                
                config = {"subset_distribution": pickle.dumps(serialized_shares),
                          "custom_rpc": "handle_missing_clients"}
                fitIns = FitIns(parameters=self.global_parameters, config=config)
                futures.append(executor.submit(client_proxy.fit, fitIns, consts.SUBSTITUTION_TIMEOUT, server_round))
            outputs = [f.result() for f in futures]

        substituted_parameters_fitRes = self.aggregate_substitution_parameters(outputs)
        results.append((None, substituted_parameters_fitRes))

    
    def _compute_substitution_shares(self, active_cids, active_client_proxies, actual_dropped, server_round):

        dropped_demand = {}
        for cid in actual_dropped:
            dist = self.label_distribution.get(cid, {})
            for label, count_enc in dist.items():
                if label not in dropped_demand:
                    dropped_demand[label] = count_enc.copy()
                else:
                    dropped_demand[label] += count_enc
        
        print(f"Dropped Demand Keys: {list(dropped_demand.keys())}")
        log_debug(f"Dropped Demand Keys: {list(dropped_demand.keys())}")

        active_stock = {}
        for cid in active_cids:
            cid_key = self.cid_to_partition.get(cid, cid)
            dist = self.label_distribution.get(str(cid_key), {})
            if not dist: dist = self.label_distribution.get(cid, {})
            
            for label, count_enc in dist.items():
                if label not in active_stock:
                    active_stock[label] = count_enc.copy()
                else:
                    active_stock[label] += count_enc


        helper_proxy = active_client_proxies[0]
        
        k_min = 0.0
        k_max = 1.0
        iterations = 5 
        
        print(f"Starting Blind Binary Search for Scaling Factor (5 iterations)...")
        
        for i in range(iterations):
            k_mid = (k_min + k_max) / 2.0
            
            blinded_checks = {}
            labels_to_check = list(dropped_demand.keys())
            
            k_enc = ts.ckks_vector(self.context, [k_mid])
            mask_val = random.uniform(10, 100)
            mask_enc = ts.ckks_vector(self.context, [mask_val])
            zero_enc = ts.ckks_vector(self.context, [0])

            for label in labels_to_check:
                if label in active_stock:
                    stock_total = active_stock[label]
                else:
                    stock_total = zero_enc # No stock means 0
                
                dropped_total = dropped_demand[label]
                
                # Active - (Dropped * k)
                diff = stock_total - (dropped_total * k_enc)
                blinded = diff * mask_enc
                blinded_checks[label] = blinded.serialize()
            
            if not blinded_checks: # Should not happen if dropped_demand is not empty
                 k_min = 1.0; k_max = 1.0; break

            ins = EvaluateIns(
                 parameters=self.global_parameters,
                 config={"check_global_feasibility": pickle.dumps(blinded_checks)}
            )
            
            res = None
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(helper_proxy.evaluate, ins, consts.SUBSTITUTION_TIMEOUT, server_round)
                    res = future.result()
            except Exception as e:
                print(f"Optimization failed: {e}")
                k_min = k_mid # Fail safe? Or break?
                break
                
            is_feasible = res.metrics.get("is_feasible", False)
            
            if is_feasible:
                k_min = k_mid # Try higher k
            else:
                k_max = k_mid # Too high, lower k
                
        final_k = k_min
        print(f"Converged Scaling Factor: {final_k:.4f}")
        
        k_final_enc = ts.ckks_vector(self.context, [final_k])
        
        scaled_dropped_demand = {}
        for label, val in dropped_demand.items():
            scaled_dropped_demand[label] = val * k_final_enc
            
        original_dropped_demand = dropped_demand
        dropped_demand = scaled_dropped_demand
        final_shares = {cid: {} for cid in active_cids} 
        
        remaining_demand = dropped_demand.copy()
        
        all_labels = set(remaining_demand.keys())
        active_set = {label: list(active_client_proxies) for label in all_labels}

        for i_loop in range(3):
            print(f"--- Protocol Iteration {i_loop + 1} ---")
            log_debug(f"--- Protocol Iteration {i_loop + 1} ---")
            
            blinded_checks_per_client = {cid: {} for cid in active_cids}
            client_proxy_map = {p.cid: p for p in active_client_proxies}
            
            labels_to_check = []
            
            for label in all_labels:
                if not active_set[label]: continue # No one left to ask
                
                # Calculate Fair Share (Encrypted)
                target = remaining_demand[label]
                num_active = len(active_set[label])
                
                # Use Vector Mult for stability
                if num_active not in self.inv_num_cache:
                    self.inv_num_cache[num_active] = ts.ckks_vector(self.context, [1.0 / num_active])
                inv_num_enc = self.inv_num_cache[num_active]
                fair_share = target * inv_num_enc
                
                for client in active_set[label]:
                    cid = client.cid
                    cid_key = self.cid_to_partition.get(cid, cid)
                    dist = self.label_distribution.get(str(cid_key), {})
                    if not dist:
                         # Try fallback
                         dist = self.label_distribution.get(cid, {})

                    stock_enc = dist.get(label)
                    
                    if stock_enc is None:
                         # Assume 0 if unknown label
                         stock_enc = ts.ckks_vector(self.context, [0]) # Should encrypt 0
                    else:
                         stock_enc = stock_enc.copy() # CRITICAL: Copy to prevent in-place modulus switching degradation

                    
                    # Generate Random Mask
                    mask_val = random.uniform(10, 100) # Arbitrary positive mask
                    
                    # Blinded Diff = (Fair_Share - Stock) * Encrypted(Mask)
                    mask_enc = ts.ckks_vector(self.context, [mask_val])
                    blinded_diff = (fair_share - stock_enc) * mask_enc 
                    
                    blinded_checks_per_client[cid][label] = blinded_diff.serialize()
                    labels_to_check.append(label)

            if not labels_to_check:
                break # Done

            # Send requests (Parallel)
            with ThreadPoolExecutor() as executor:
                futures = {}
                for cid, checks in blinded_checks_per_client.items():
                    if not checks: continue
                    
                    # Encrypted values are already serialized
                    serialized_checks = checks
                    
                    ins = EvaluateIns(
                         parameters=self.global_parameters, # Not used but required
                         config={"blinded_diff": pickle.dumps(serialized_checks)}
                    )
                    futures[cid] = executor.submit(
                        client_proxy_map[cid].evaluate, ins, consts.SUBSTITUTION_TIMEOUT, server_round
                    )
                
                # Collect Responses
                for cid, future in futures.items():
                    res = future.result() # EvaluateRes
                    res = future.result() # EvaluateRes
                    # If execution failed, future.result() would have raised exception
                    metrics = res.metrics
                    is_capped_map = pickle.loads(metrics["is_capped"])
                    
                    for label, is_capped in is_capped_map.items():
                        if is_capped:
                             cid_key = self.cid_to_partition.get(cid, cid)
                             dist = self.label_distribution.get(str(cid_key), {})
                             if not dist:
                                 dist = self.label_distribution.get(cid, {})
                             stock = dist.get(label, ts.ckks_vector(self.context, [0]))
                             final_shares[cid][label] = stock
                             
                             remaining_demand[label] -= stock
                             
                             proxy = client_proxy_map[cid]
                             if proxy in active_set[label]:
                                 active_set[label].remove(proxy)
                        else:
                             pass

        for label in all_labels:
            active_clients = active_set[label]
            if not active_clients: continue
            
            target = remaining_demand[label]
            fair_share = target * (1.0 / len(active_clients))
            
            for client in active_clients:
                 final_shares[client.cid][label] = fair_share

        return final_shares



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