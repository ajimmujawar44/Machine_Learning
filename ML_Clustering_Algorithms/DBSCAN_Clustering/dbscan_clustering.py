# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:05:15 2026

@author: Lenovo
"""

# ==========================================
# DBSCAN Clustering
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv(r"E:\ML\Machine learning\ML_Clustering_Algorithms\data\Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values

# ==========================================
# Feature Scaling
# ==========================================

sc = StandardScaler()

X_scaled = sc.fit_transform(X)

# ==========================================
# Training DBSCAN
# ==========================================

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5,
    metric='euclidean'
)

y_dbscan = dbscan.fit_predict(X_scaled)

# ==========================================
# Save Output
# ==========================================

dataset["Cluster"] = y_dbscan

dataset.to_csv("Mall_Customers_DBSCAN.csv",index=False)

print(os.getcwd())

print("CSV Saved Successfully")

# ==========================================
# Visualization
# ==========================================

plt.figure(figsize=(10,7))

clusters = set(y_dbscan)

for cluster in clusters:

    if cluster==-1:

        plt.scatter(
            X[y_dbscan==-1,0],
            X[y_dbscan==-1,1],
            s=80,
            c='black',
            label='Noise'
        )

    else:

        plt.scatter(
            X[y_dbscan==cluster,0],
            X[y_dbscan==cluster,1],
            s=80,
            label=f'Cluster {cluster+1}'
        )

plt.title("DBSCAN Clustering")

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.legend()

plt.grid(True)

plt.show()