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
        self.previous_round_updates: dict[str, list[np.ndarray]] = {}

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

        # 1. Capture updates for future substitution (Cosine Similarity)
        # Store flattened update vectors (gradient = w_prime - w_global)
        import numpy as np
        
        # Helper to flatten parameters
        def flatten_params(params):
            return np.concatenate([p.flatten() for p in params])

        # Convert global parameters to numpy once
        if self.global_parameters:
             from flwr.common import parameters_to_ndarrays
             current_global_np = parameters_to_ndarrays(self.global_parameters)
             
             for client, fit_res in results:
                 # Check if the client actually has parameters (failures might be in results?)
                 # The type signature says (ClientProxy, FitRes). Failures are separate. 
                 # But let's be safe.
                 if client:
                     if fit_res.parameters:
                         client_params = parameters_to_ndarrays(fit_res.parameters)
                         # Calculate update: delta = new - old
                         update_vector = flatten_params([c - g for c, g in zip(client_params, current_global_np)])
                         self.previous_round_updates[client.cid] = update_vector

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
        """Substitute dropped clients with subset training using Per-Client Substitution."""
        if (len(failures) == 0):
            print(f"No dropped clients to substitute in round {server_round}.")
            return
        
        # 1. Identify Dropped Clients
        active_cids = [x[0].cid for x in results]
        all_known_cids = list(self.label_distribution.keys())
        dropped_cids = list(set(all_known_cids) - set(active_cids))

        if not dropped_cids:
             print("No known dropped clients to substitute (based on label_distribution history).")
             return

        print(f"Total Dropped Clients: {len(dropped_cids)}")
        
        # 2. Iterate Over EACH Dropped Client
        for dropped_cid in dropped_cids:
            print(f"Generating substitute for Dropped Client {dropped_cid}...")
            
            # Select Top-K Clients based on Capability (Label Overlap)
            # Returns list of (cid, score)
            selected_active_clients_with_scores = self.get_top_k_capability_clients(
                dropped_cid, 
                active_cids, 
                k=consts.TOP_K_CLIENTS
            )
            
            selected_active_cids = [cid for cid, _ in selected_active_clients_with_scores]
            similarity_scores = {cid: score for cid, score in selected_active_clients_with_scores}
            
            print(f"Selected {len(selected_active_cids)} clients for substitution: {selected_active_cids}")
            
            # Target is just this one dropped client
            target_dropped_list = [dropped_cid]
            
            # Distribute work to SELECTED active clients
            subset_dist_per_client = self.get_subset_distribution_for_active_clients(
                selected_active_cids, 
                target_dropped_list
            )
            
            
            # Correct approach for weighted aggregation:
            # We need to map `fitRes` back to `cid` to find its `similarity_score`.
            # Since `client_proxy` has `cid`, we can track it.
            
            client_fit_results = [] # List[(cid, FitRes)]
            with ThreadPoolExecutor(max_workers=consts.MAX_CONCURRENT_SUBSTITUTIONS) as executor:
                future_to_cid = {}
                for client_proxy, _ in results:
                     if client_proxy and client_proxy.cid in subset_dist_per_client and subset_dist_per_client[client_proxy.cid]:
                        subset_distribution_bytes = pickle.dumps(subset_dist_per_client[client_proxy.cid])
                        config = {"subset_distribution": subset_distribution_bytes,
                                  "custom_rpc": "handle_missing_clients"}
                        fitIns = FitIns(parameters=self.global_parameters, config=config)
                        fut = executor.submit(client_proxy.fit, fitIns, consts.SUBSTITUTION_TIMEOUT, server_round)
                        future_to_cid[fut] = client_proxy.cid
                
                for fut in future_to_cid:
                    try:
                        res = fut.result()
                        cid = future_to_cid[fut]
                        client_fit_results.append((cid, res))
                    except Exception as e:
                        print(f"Substitution task for client {future_to_cid[fut]} failed: {e}")

            if client_fit_results:
                # Aggregate to form the single substitute for THIS dropped client
                substitute_update = self.aggregate_substitution_parameters_weighted(client_fit_results, similarity_scores)
                results.append((None, substitute_update))


    def consolidate_label_distributions(self, active_clients_ids, dropped_clients_ids):
        # dropped_clients_ids is now explicit
        
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

        return dropped_clients_distribution, active_clients_distribution
    
    
    def get_consolidated_representative_distribution(self, dropped_clients_distribution, active_clients_distribution):
        # Calculate the representative subset distribution
        representative_subset_distribution = {}
        total_dropped = sum(dropped_clients_distribution.values())

        if total_dropped == 0:
             return {}

        target_percentages = {
            label: count / total_dropped
            for label, count in dropped_clients_distribution.items()
        }

        # Handle case where active clients have NO common labels with dropped
        common_labels = set(dropped_clients_distribution.keys()) & set(active_clients_distribution.keys())
        if not common_labels:
            print("Warning: No overlap between active and dropped labels in this cluster.")
            return {}

        anchor_label = max(
            common_labels,
            key=lambda label: dropped_clients_distribution[label],
        )

        representative_subset_distribution[anchor_label] = min(
            active_clients_distribution[anchor_label],
            dropped_clients_distribution[anchor_label])
        
        # Calculate the total count based on dropped percentage and chosen anchor label value
        if target_percentages[anchor_label] > 0:
            anchor_label_total = floor(representative_subset_distribution[anchor_label] / target_percentages[anchor_label])
        else:
            anchor_label_total = 0

        for label, count in active_clients_distribution.items():
            if label == anchor_label:
                continue
            # If label was present in dropped
            if label in target_percentages:
                target_count = floor(target_percentages[label] * anchor_label_total)
                representative_subset_distribution[label] = min(target_count, count)        

        # print(f"Subset distribution for active clients: {representative_subset_distribution}")
        return representative_subset_distribution
    
    
    def get_subset_distribution_for_active_clients(
            self,
            active_clients_ids: list[str],
            dropped_clients_ids: list[str]) -> dict:
        """Get representative subset distribution for active clients in a cluster."""
        
        dropped_clients_distribution, active_clients_distribution = self.consolidate_label_distributions(active_clients_ids, dropped_clients_ids)

        print(f"Dropped clients distribution: {dropped_clients_distribution}")
        print(f"Active clients distribution: {active_clients_distribution}")

        representative_subset_distribution = self.get_consolidated_representative_distribution(
            dropped_clients_distribution, active_clients_distribution)   

        print(f"Representative subset distribution: {representative_subset_distribution}")

        # Distribute the representative subset among active clients
        subset_distribution_per_client = {cid: {} for cid in active_clients_ids}
        
        for label, total_needed in representative_subset_distribution.items():
            client_counts = [
                (cid, self.label_distribution.get(cid, {}).get(label, 0))
                for cid in active_clients_ids
            ]
            idx = 0
            # Safety: avoid infinite loop if no client has the label (though min() logic above should prevent this)
            loop_limit = len(client_counts) * total_needed + 100 
            loop_count = 0
            
            while total_needed > 0 and any(count > 0 for _, count in client_counts) and loop_count < loop_limit:
                cid, available = client_counts[idx % len(client_counts)]
                if available > 0:
                    subset_distribution_per_client[cid][label] = subset_distribution_per_client[cid].get(label, 0) + 1
                    client_counts[idx % len(client_counts)] = (cid, available - 1)
                    total_needed -= 1
                idx += 1
                loop_count += 1
                
        return subset_distribution_per_client
    
    
    def aggregate_substitution_parameters_weighted(self, results: list[tuple[str, FitRes]], similarity_scores: dict[str, float]) -> FitRes:
        """
        Aggregates results using Cosine Similarity scores as weights.
        results: List of (cid, FitRes)
        similarity_scores: Dict mapping cid to score
        """
        if not results:
             return None

        # Prepare weights
        # weight_i = num_examples_i * max(0, similarity_score_i)
        # We ensure score is non-negative for weight (though cosine can be negative). 
        # Negative similarity means opposite direction -> maybe we should flip gradient? 
        # For now, let's clip at 0 or use just max(0, score).
        
        weighted_weights = []
        total_examples = 0
        
        from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
        
        for cid, fit_res in results:
            score = max(0.001, similarity_scores.get(cid, 0.0)) # Avoid 0 division or vanishing
            weight_scale = fit_res.num_examples * score
            
            params = parameters_to_ndarrays(fit_res.parameters)
            weighted_weights.append((params, weight_scale))
            total_examples += fit_res.num_examples # Keep track of 'real' samples for accounting?
            
        # Weighted Average
        # Sum(w * weight) / Sum(weight)
        
        if not weighted_weights:
            return None
            
        # Initialize accumulator with first element
        first_params, first_weight = weighted_weights[0]
        accumulated = [p * first_weight for p in first_params]
        total_weight_scale = first_weight
        
        for params, weight in weighted_weights[1:]:
            total_weight_scale += weight
            for i, p in enumerate(params):
                accumulated[i] += p * weight
                
        # Normalize
        if total_weight_scale > 0:
            final_params = [p / total_weight_scale for p in accumulated]
        else:
            final_params = first_params # Fallback
            
        aggregated_results = FitRes(
            parameters=ndarrays_to_parameters(final_params),
            num_examples=total_examples,
            metrics={},
            status=None
        )
        return aggregated_results

        
        return aggregated_results

    def get_top_k_relevant_clients(self, dropped_cid: str, active_cids: list[str], k: int) -> list[str]:
        """
        [Fallback Strategy]
        Selects the top K active clients that are most relevant to the dropped client.
        Relevance is defined as the sum of overlapping label counts.
        """
        target_dist = self.label_distribution.get(dropped_cid, {})
        
        if not target_dist:
            return []

        scored_clients = []
        for cid in active_cids:
            client_dist = self.label_distribution.get(cid, {})
            score = 0
            # Calculate relevance score: intersection of distributions
            for label, count in client_dist.items():
                if label in target_dist:
                    score += min(count, target_dist[label])
            
            scored_clients.append((score, cid))
            
        # Sort by score descending
        scored_clients.sort(key=lambda x: x[0], reverse=True)
        
        # Pick top K
        top_k = [cid for score, cid in scored_clients[:k] if score > 0]
        return top_k

    def get_top_k_capability_clients(self, dropped_cid: str, active_cids: list[str], k: int) -> list[tuple[str, float]]:
        """
        Selects top K active clients based on their capability to substitute the dropped client's data.
        Capability score is defined as the sum of overlapping label counts: score = sum(min(dropped[l], active[l]))
        """
        dropped_dist = self.label_distribution.get(dropped_cid, {})
        
        if not dropped_dist:
            print(f"Warning: No label distribution for dropped client {dropped_cid}. Returning random active clients.")
            return [(cid, 1.0) for cid in active_cids[:k]]

        scored_clients = []
        for cid in active_cids:
            active_dist = self.label_distribution.get(cid, {})
            score = 0.0
            
            # Calculate overlap capability score
            for label, dropped_count in dropped_dist.items():
                active_count = active_dist.get(label, 0)
                score += min(dropped_count, active_count)
            
            scored_clients.append((score, cid))
            
        # Sort by score descending
        scored_clients.sort(key=lambda x: x[0], reverse=True)
        
        # Format as (cid, score) list
        top_k = [(cid, score) for score, cid in scored_clients[:k]]
        
        # Log the scores for debugging
        print(f"Capability Scores for {dropped_cid}: {[(cid, f'{score:.1f}') for cid, score in top_k]}")
        
        return top_k

    def get_top_k_similarity_clients(self, dropped_cid: str, active_cids: list[str], k: int) -> list[str]:
        """
        Selects top K active clients by Cosine Similarity of their previous round's weight updates.
        Falls back to Label Relevance if no history exists for the dropped client.
        """
        # Check if we have history
        if dropped_cid not in self.previous_round_updates:
            print(f"Warning: No update history for client {dropped_cid}. Fallback to Label Relevance.")
            # Map fallback list to list of tuples with dummy score 1.0
            fallback_list = self.get_top_k_relevant_clients(dropped_cid, active_cids, k)
            return [(cid, 1.0) for cid in fallback_list]
            
        target_update = self.previous_round_updates[dropped_cid]
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        
        scored_clients = []
        
        # Target needs to be 2D for sklearn: (1, n_features)
        target_vector = target_update.reshape(1, -1)
        
        for cid in active_cids:
            if cid in self.previous_round_updates:
                candidate_vector = self.previous_round_updates[cid].reshape(1, -1)
                sim = cosine_similarity(target_vector, candidate_vector)[0][0]
                scored_clients.append((sim, cid))
            else:
                 # If active client has no history (new?), give it low score?
                 # Or treat as 0 similarity
                 scored_clients.append((-1.0, cid))
                 
        # Sort by similarity descending
        scored_clients.sort(key=lambda x: x[0], reverse=True)
        
        top_k = [(cid, sim) for sim, cid in scored_clients[:k]]
        print(f"Cosine Similarities for {dropped_cid}: {[(cid, f'{sim:.4f}') for cid, sim in top_k]}")
        
        return top_k