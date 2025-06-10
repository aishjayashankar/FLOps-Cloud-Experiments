from flwr.server.strategy.aggregate import aggregate_inplace
from flwr.common import FitRes, Parameters, ndarrays_to_parameters

def aggregate(parameters):
    results = []
    num_samples = [1555, 899, 846, 580]
    idx = 0
    print("Starting aggregation of parameters...")
    for param in parameters:
        print(f"Aggregating parameters for client {idx} with {num_samples[idx]} samples.")
        fitRes = FitRes(
            parameters=ndarrays_to_parameters(param),
            num_examples=num_samples[idx],
            metrics=None,
            status=None
        )
        idx += 1
        results.append((None, fitRes))
    print("All client parameters collected. Performing in-place aggregation...")
    aggregated = aggregate_inplace(results)
    print("Aggregation complete.")
    return aggregated