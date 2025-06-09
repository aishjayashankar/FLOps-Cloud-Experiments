import flops_infra_drift.consts as consts
import pickle

from flwr.common import FitRes, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate_inplace

def aggregate(parameters):
    results = []
    idx = 0
    print("Starting aggregation of parameters...")
    for param in parameters:
        print(f"Aggregating parameters for client {idx} with {consts.SAMPLES_COUNT_LIST[idx]} samples.")
        fitRes = FitRes(
            parameters=ndarrays_to_parameters(param),
            num_examples=consts.SAMPLES_COUNT_LIST[idx],
            metrics=None,
            status=None
        )
        idx += 1
        results.append((None, fitRes))
    print("All client parameters collected. Performing in-place aggregation...")
    aggregated = aggregate_inplace(results)
    print("Aggregation complete.")
    return aggregated

def get_dropped_client_parameters(results: list[tuple[ClientProxy, FitRes]]) -> tuple[ClientProxy, FitRes]:
    # Get list of substituted parameters from results
    print("Extracting dropped client parameters from results...")
    substituted_parameters = []
    for _, fitRes in results:
        parameters_bytes = fitRes.metrics.get("dropped_client_parameters_bytes")
        if parameters_bytes is not None:
            dropped_client_parameters = pickle.loads(parameters_bytes)
            substituted_parameters.append(dropped_client_parameters)
    print(f"Extracted {len(substituted_parameters)} sets of substituted parameters.")

    # Aggregate the substituted parameters
    aggregated_parameters = aggregate(substituted_parameters)
    
    # Client 4 sample size: 12960
    fitRes = FitRes(
        parameters=ndarrays_to_parameters(aggregated_parameters),
        num_examples=consts.SUBSTITUTION_SAMPLE_SIZE,
        metrics=None,
        status=None
    ) 

    results.append((None, fitRes))
    print("Appended substituted parameters to results.")