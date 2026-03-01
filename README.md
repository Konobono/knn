# k-Nearest Neighbors (k-NN) Classifier

##  Overview

This repository contains a Python implementation of the **k-Nearest Neighbors (k-NN)** classification algorithm from scratch - another academic project.  
The goal is to demonstrate understanding of a classical machine learning method and evaluate its performance on a real dataset.  
This implementation does **not** rely on high-level machine learning libraries (e.g., scikit-learn for the classifier itself), but instead manually computes distances and predicts labels based on nearest neighbors in the training set.

##  Dataset

- The dataset used is stored in `vine_final.csv`.  
- Each row represents one instance with numerical features and a target label in the last column.  
- The data is first shuffled and then split into **training (60%)**, **validation (20%)**, and **testing (20%)** subsets.

## 🧪 Requirements

Install necessary Python packages:

```bash
pip install pandas numpy
