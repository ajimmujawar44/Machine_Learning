# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:17:01 2026

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 2026

@author: Lenovo
"""

# ==========================================
# Gaussian Mixture Model (GMM)
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ==========================================
# Import Dataset
# ==========================================

dataset = pd.read_csv(r"E:\ML\Machine learning\ML_Clustering_Algorithms\data\Mall_Customers.csv")

# Annual Income & Spending Score
X = dataset.iloc[:, [3, 4]].values

# ==========================================
# Feature Scaling
# ==========================================

sc = StandardScaler()

X_scaled = sc.fit_transform(X)

# ==========================================
# Train Gaussian Mixture Model
# ==========================================

gmm = GaussianMixture(
    n_components=5,
    covariance_type='full',
    random_state=42
)

y_gmm = gmm.fit_predict(X_scaled)

# ==========================================
# Add Cluster Column
# ==========================================

dataset["Cluster"] = y_gmm

# ==========================================
# Save Dataset
# ==========================================

dataset.to_csv("Mall_Customers_GMM.csv", index=False)

print("CSV File Saved Successfully")

print("\nCurrent Working Directory")

print(os.getcwd())

# ==========================================
# Display First 10 Records
# ==========================================

print("\nFirst 10 Records")

print(dataset.head(10))

# ==========================================
# Number of Customers in Each Cluster
# ==========================================

print("\nCustomers in Each Cluster")

print(dataset["Cluster"].value_counts().sort_index())

# ==========================================
# Cluster Summary
# ==========================================

summary = dataset.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print("\nCluster Summary")

print(summary)

summary.to_csv("Cluster_Summary_GMM.csv")

# ==========================================
# Visualization
# ==========================================

plt.figure(figsize=(10,7))

for i in range(5):

    plt.scatter(
        X[y_gmm==i,0],
        X[y_gmm==i,1],
        s=80,
        label=f"Cluster {i+1}"
    )

plt.title("Gaussian Mixture Model Clustering")

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.legend()

plt.grid(True)

plt.show()

# ==========================================
# Cluster Distribution
# ==========================================

dataset["Cluster"].value_counts().sort_index().plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Customers in Each Cluster")

plt.xlabel("Cluster")

plt.ylabel("Number of Customers")

plt.grid(True)

plt.show()