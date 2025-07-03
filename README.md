# DIWS: Distribution-Informed Weight Substitution for Infrastructure Drift in Federated Learning

This repository contains the implementation of **DIWS (Distribution-Informed Weight Substitution)**, a lightweight and privacy-preserving method for handling client dropouts in Federated Learning (FL). DIWS enables robust model training by approximating the statistical contribution of unavailable clients using label distribution metadata.

## Key Features

- **Dropout Recovery**: Approximates missing client updates using label distributions.
- **Privacy-Preserving**: No raw data sharing. Uses only pre-shared metadata.
- **Compatible**: Seamlessly integrates with standard FL Flower framework.
- **Scalable**: Designed to support real-world FL environments with heterogeneous and unreliable clients.

## Core Idea

DIWS leverages pre-shared label distribution metadata from clients. When one or more clients drop out during training, DIWS aggregates their label distributions and sends a target distribution to the remaining active clients. These clients locally generate representative updates to fill in for the dropped ones, maintaining training continuity and convergence.

## Components of the Implementation

- **client_app**: Runs on FL clients, modified to simulate dropping of clients as well as incorporate weight substitution.
- **server_app**: Driver which runs on the FL server, initializes and orchestrates the run.
- **task**: Does the heavy lifting of ML component such as loading data, model, training and evaluation.
- **diws**: Runs on FL server. Novel framework to address dropping of weights via Distribution-Informed Weight Substitution.
- **subset_client_trainer**: Runs on FL clients, takes input from 'diws' module and performs training of representative data for weight substitution.

## How to use this Repository?

Flower FL setup is required to run the programs. Can be on-premise nodes, cloud VMs or simulated nodes.

The DIWS module acts as a wrapper around the aggregation algorithm used for the run. Refer to server_app.py's initialization of aggregation strategy for details.

To simulate node dropping behaviour the consts.py file can be modified. It includes the following configurable variables:

- CLIENT_DROP_ROUND_START: An integer that determines the round at which node drops during the run
- CLIENT_DROP_ROUND_END: An integer that determines the round at which the node rejoins the run
- DROPPED_CLIENT_PARITIONS_IDS: Array denoting the clients that will be dropping out

For weight substitution, the DIWS module performs the necessary calculations and then invokes the client_app's fit method with config["custom_rpc"] set to "handle_missing_clients". To incorporate that the client_app needs to be modified to invoke the subset_client_trainer, refer client_app.py's fit method for details.

Note that the example used in the repository is based on CIFAR-10 dataset. If you wish to use any other dataset then the subset_client_trainer needs to be modified accordingly. Mainly, the parameters and label handling needs to change as per the dataset.
