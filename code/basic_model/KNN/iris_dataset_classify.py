#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Iris Dataset Classification

第一次接触机器学习
"""

__time__ = '2025-09-19'

# Step1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import os

# Step2: Load dataset
# loading Iris dataset
iris = load_iris()

# Step3: Data preprocessing
# change data into pandas DataFrame
X = pd.DataFrame(iris.data, columns=iris.feature_names) # feature
Y = pd.DataFrame(iris.target) # target

# data explore - show first 5 rows
print(X.head())

# devide training dataset and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# standardize features
# Standardize after deviding: Avoid data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step4: Model selection and training
# use KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# train model
knn.fit(X_train, Y_train)

# Step5: Model evaluation
# predict on test set
Y_pred = knn.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step6: Visualization
# 2-Dimensional plot
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred, cmap='viridis', marker='o')
plt.title("KNN Classification Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Save plot
output_dir = 'code/basic_model/KNN/plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'iris_knn_results.png'),
            dpi=300, bbox_inches='tight')
print(f"Plot saved to {os.path.join(output_dir, 'iris_knn_results.png')}")