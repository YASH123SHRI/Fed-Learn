# Fed-Learn

# Federated Learning in Cybersecurity

This project investigates the integration of **Federated Learning (FL)** and **Differential Privacy (DP)** to improve cybersecurity solutions like intrusion and malware detection while ensuring data confidentiality. It demonstrates how decentralized machine learning can be effective even in low-resource environments without compromising privacy.

## Project Overview

Traditional centralized machine learning systems require raw data transmission, which can pose privacy risks. Our project solves this by:

* Training local models on decentralized cybersecurity datasets.
* Aggregating model updates on a central server using **Federated Averaging (FedAvg)**.
* Applying **Differential Privacy** to model updates to protect sensitive client information.
* Using **SSL/TLS** for secure communication.
* Ensuring performance remains close to centralized models even on college-standard hardware.

##  Key Features

* Decentralized training on multiple simulated clients.
* Differential Privacy integration for secure updates.
* Logging system for anomaly detection and training progress.
* Low-end hardware compatibility for broader accessibility.
* Evaluation metrics for accuracy, convergence, and communication efficiency.
* Version control & reproducibility enabled for future experimentation.

##  Use Cases

* Intrusion Detection Systems (IDS)
* Malware Detection
* Privacy-preserving threat intelligence sharing
* Real-time cyber defense using edge devices

##  Functional Requirements

* Local model training on decentralized cybersecurity datasets
* Encrypted model update aggregation on the server
* Logging mechanism to monitor training and anomalies

##  Non-Functional Requirements

* Usable on standard college laptops
* Accuracy within ±10% of centralized models
* Simulated environment for 5–10 clients
* Reproducible with proper documentation and version control

##  Technologies Used

* Python
* PyTorch / TensorFlow (choose based on implementation)
* NumPy, Pandas, Scikit-learn
* SSL/TLS for secure communication
* Matplotlib/Seaborn for visualization
