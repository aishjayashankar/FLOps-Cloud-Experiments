# DIWS: Distribution-Informed Weight Substitution for Infrastructure Drift in Federated Learning

This repository contains the implementation of **DIWS (Data-Informed Weight Substitution)**, a lightweight and privacy-preserving method for handling client dropouts in Federated Learning (FL). DIWS enables robust model training by approximating the statistical contribution of unavailable clients using label distribution metadata.

## Key Features

- **Dropout Recovery**: Approximates missing client updates using label distributions.
- **Privacy-Preserving**: No raw data sharing—uses only pre-shared metadata.
- **Compatible**: Seamlessly integrates with standard FL workflows (e.g., Flower).
- **Scalable**: Designed to support real-world FL environments with heterogeneous and unreliable clients.

## Core Idea

DIWS leverages pre-shared label distribution metadata from clients. When one or more clients drop out during training, DIWS aggregates their label distributions and sends a target distribution to the remaining active clients. These clients locally generate representative updates to fill in for the dropped ones, maintaining training continuity and convergence.

