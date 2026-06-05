# Modeling Urban Disease Vulnerability: Epidemic Threshold Prediction via Synthetic Mobility Graphs and Graph Neural Networks

This repository contains the codebase for generating synthetic mobility networks, calculating analytical epidemic thresholds, and training an Edge-Conditioned Graph Neural Network (ECC-GNN) to serve as a data-efficient computational surrogate.

## 📁 Repository Structure

```text
├── data_generation/
│   ├── datageneration_pipeline.py  # Generates gravity/radiation models & analytical thresholds
│   ├── covid_timeline_processing.py # Processes Georgia COVID-19 timeline & case data
│   └── data_processing.py           # General demographic and empirical data parsing
├── datasets/
│   ├── empirical/                  # Raw/processed London, Georgia, and Japan networks
│   └── sparsity_levels/            # Graphs configured for different data sparsity regimes
├── gnn_training.py                 # Core script to train the proposed GNN and baselines
└── README.md
