# FLOps Cloud Experiments

## Overview

This repository outlines the Azure-VM based experiments using Flower framework, for the FLOps Infrastructure Drift project.

## Experimental Testbed

- 5 client VMs and 1 server VM
    - Heterogeneous compute SKUs (D2s_v3, D2as_v4, D2ads_v6, D2alds_v6)
- 50 rounds of FL training 
- Datasets used: CIFAR-10 and FMNIST
- Models: MobNetv3-Small and ResNet18
- Aggregation Strategies: FedAvg and FedProx

## Experiments Run

- Baselines (Branches: `dev/base-*`)
- Scenario 1: Node disconnection and rejoining (Branches: `dev/ndc-*`)
    - Nodes 3, 4, 5 disconnect at rounds 5, 6, 7 respectively
    - All nodes rejoin at round 31
- Scenario 2: Timeout for straggler nodes (Branches: `dev/timeout-*`)
    - Different timeouts used based on average straggler node training duration
- Scenario 3: TBD

## Experiment Results

The plots and script used for generating them are available in the `RunArtifacts` folder in the `main` branch.

All the log files pertaining to the runs are available at [FLOps Experiment Database - Notion](https://www.notion.so/dash-lab/19c8db185a758082a8e9f78dea0d1cb6?v=19c8db185a75803ebb7d000cd0ba3635)