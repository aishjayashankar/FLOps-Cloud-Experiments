from centralized_client import client_fn
from centralized_aggregator import aggregate
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# logger.info("Initializing clients")
# clients = [client_fn(i) for i in range(0, 4)]

def load_parameters_from_file(round_number):
    folder_path = "/mnt/Users/Ketan/Desktop/PhD/FLOpsInfraDrift/RunArtifacts/ConsolidatedData/Parameters/Aggregated/"
    file_name = f"aggregated-parameters-round-{round_number}.csv"
    params_file = folder_path + file_name
    logger.info(f"Reading parameters from {params_file}")
    params_df = pd.read_csv(params_file, header=None)
    params_df = params_df.drop(0)  # Drop the first row (column headers)
    logger.info("Converting parameters to NumPy arrays")
    return [np.array(eval(v)) for v in params_df.iloc[:, 1]]

# params = load_parameters_from_file(6)

# for round_idx in range(7, 31):
#     updated_params_list = []
#     client_num = 0
#     for client in clients:
#         logger.info(f"Round {round_idx}: Fitting for client {client_num}")
#         updated_params, _, _ = client.fit(params)
#         updated_params_list.append(updated_params)
#         client_num += 1

#     params = aggregate(updated_params_list)

#     logger.info(f"Round {round_idx}: Evaluating aggregated parameters")
#     loss, dataset_len, accuracy = clients[0].evaluate(params)
#     print(f"Dataset Length = {dataset_len}, Loss = {loss}, Accuracy = {accuracy['accuracy']}")

print("Initializing client")
client = client_fn(0)
params = load_parameters_from_file(6)

for round_idx in range(7, 31):
    logger.info(f"Round {round_idx}: Fitting for client")
    updated_params, _, _ = client.fit(params)
    params = updated_params
    logger.info(f"Round {round_idx}: Evaluating fit parameters")
    loss, dataset_len, accuracy = client.evaluate(params)
    print(f"Dataset Length = {dataset_len}, Loss = {loss}, Accuracy = {accuracy['accuracy']}")
