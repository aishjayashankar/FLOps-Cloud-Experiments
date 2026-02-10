# Addressing Client Dropout in Federated Learning via Distribution-Informed Weight Substitution

This repository serves as the official implementation of the **Distribution-Informed Weight Substitution (DIWS)** framework, a novel approach designed to mitigate the effects of client dropout in Federated Learning (FL).

## Core Concept

Infrastructure drift, specifically **client dropout**, poses a significant challenge in Federated Learning, leading to stale models and degraded accuracy. Traditional methods for handling dropout often struggle with prolonged absences or rely on limited substitution strategies that compromise robustness.

**DIWS** addresses these limitations by substituting the missing updates of dropped clients with information derived from the shared label distribution of active clients. 

**Key Features:**
*   **Robust Performance**: Effectively maintains global model accuracy even when a significant number of clients drop out.
*   **Adaptive & Fresh**: Substitution weights are recalculated at each training round, ensuring model parameters remain relevant even during extended dropout periods.
*   **Privacy-Preserving**: Integrates **CKKS-based Fully Homomorphic Encryption (FHE)** to secure label metadata, ensuring compliance with strict privacy standards against both the server and adversarial clients.
*   **Scalable Architecture**: Utilizes a clustering-based approach to handle substitutions locally, minimizing computational burden and communication overhead.

## Project Structure & Setup

This project is built using the **Flower** (Flwr) framework for Federated Learning simulations.

### Prerequisites

*   **Python**: Version 3.10 or higher is recommended.
*   **Virtual Environment**: It is highly recommended to run this project within a dedicated virtual environment.
*   **NVIDIA GPU**: Required for running ML training, can be substituted with CPU equivalent, but requirements.txt needs to be modified accordingly.

### Installation

1.  **Clone the Repository** and navigate to the project directory.
2.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    Install the required packages as specified in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To execute the Federated Learning experiments, follow these steps:

1.  **Activate your virtual environment** (if not already active).
2.  **Ensure correct directory structure**:
    *   The `pyproject.toml` file **must** reside at the root of the workspace, outside of the specific source code directory (e.g., `DIWS-Implementation`). This file contains the critical configuration for the Flower application and maps the `serverapp` and `clientapp`.
3.  **Run the Simulation**:
    Execute the following command from the root directory (where `pyproject.toml` is located):
    ```bash
    flwr run .
    ```

This command initiates the simulation environment, setting up the server and clients according to the parameters defined in `[tool.flwr.app.config]`.

## Repository Structure

The codebase is organized as follows:

*   **`client_app.py`**: The client-side entry point. Defines the `FlowerClient` class responsible for local model training, evaluation, and safely sharing encrypted label distributions.
*   **`server_app.py`**: The server-side entry point. Configures the federated learning strategy and initializes the `DIWS` wrapper around the standard `FedAvg` strategy.
*   **`diws.py`**: The core implementation of the **DIWS Strategy**. This module handles the detection of dropped clients, computes substitution shares using FHE-encrypted data, and manages the aggregation process.
*   **`keys.py`**: **Privacy & Security Module**. Handles the generation and loading of **Fully Homomorphic Encryption (FHE)** keys (CKKS scheme). It generates the public context (for the server) and private context (for clients) to ensure all label distribution metadata remains encrypted and secure.
*   **`subset_client_trainer.py`**: Defines the `SubsetClientTrainer`, a specialized class used during the substitution phase. It enables training on a mathematically derived subset of data to replicate the contribution of a dropped client.
*   **`task.py`**: Contains core utility functions for the machine learning task, including data loading (CIFAR-10), standard training loops, and evaluation functions.
*   **`consts.py`**: Defines global constants such as dropout intervals, participation settings, and file paths.
*   **`pyproject.toml`**: The main configuration file. **Critical**: This file must reside at the workspace root to correctly map the `serverapp` and `clientapp` entry points for the Flower framework.

## Simulating Client Dropout

To simulate client dropout behavior, you can modify the following variables in `consts.py`:

*   **`CLIENT_DROP_ROUND_START`**: The round index at which designated clients drop out.
*   **`CLIENT_DROP_ROUND_END`**: The round index at which dropped clients rejoin the FL run.
*   **`DROPPED_CLIENT_PARTITIONS_IDS`**: A list of partition IDs identifying the clients that will drop out.
