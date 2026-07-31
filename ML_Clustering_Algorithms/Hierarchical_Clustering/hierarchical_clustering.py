# ==========================================
# Hierarchical Clustering
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import os

from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv(r"E:\ML\Machine learning\ML_Clustering_Algorithms\data\Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values

# ==========================================
# Dendrogram
# ==========================================

plt.figure(figsize=(10,6))

sch.dendrogram(
    sch.linkage(X,method='ward')
)

plt.title("Dendrogram")

plt.xlabel("Customers")

plt.ylabel("Euclidean Distance")

plt.show()

# ==========================================
# Training Model
# ==========================================

hc = AgglomerativeClustering(
    n_clusters=5,
    metric='euclidean',
    linkage='ward'
)

y_hc = hc.fit_predict(X)

# ==========================================
# Save Output
# ==========================================

dataset["Cluster"] = y_hc

dataset.to_csv("Mall_Customers_Hierarchical.csv",index=False)

print(os.getcwd())

print("CSV Saved Successfully")

# ==========================================
# Visualization
# ==========================================

plt.figure(figsize=(10,7))

for i in range(5):

    plt.scatter(
        X[y_hc==i,0],
        X[y_hc==i,1],
        s=80,
        label=f"Cluster {i+1}"
    )

plt.title("Hierarchical Clustering")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.legend()

plt.grid(True)

plt.show()