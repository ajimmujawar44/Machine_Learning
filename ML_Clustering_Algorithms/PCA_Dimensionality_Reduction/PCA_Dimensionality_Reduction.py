
# ==========================================
# Principal Component Analysis (PCA)
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# Import Dataset
# ==========================================

dataset = pd.read_csv(r"E:\ML\Machine learning\ML_Clustering_Algorithms\data\Mall_Customers.csv")

# Select Numerical Features
X = dataset.iloc[:, 2:5].values
# Age, Annual Income, Spending Score

# ==========================================
# Feature Scaling
# ==========================================

sc = StandardScaler()

X_scaled = sc.fit_transform(X)

# ==========================================
# Apply PCA
# ==========================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# ==========================================
# Display Information
# ==========================================

print("Original Shape :", X.shape)

print("Reduced Shape :", X_pca.shape)

print("\nExplained Variance Ratio")

print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance")

print(sum(pca.explained_variance_ratio_))

print("\nPrincipal Components")

print(pca.components_)

# ==========================================
# Save PCA Dataset
# ==========================================

pca_dataset = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_dataset.to_csv(
    "Mall_Customers_PCA.csv",
    index=False
)

print("\nPCA Dataset Saved Successfully")

print("\nCurrent Working Directory")

print(os.getcwd())

# ==========================================
# PCA Scatter Plot
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    s=60
)

plt.title("Principal Component Analysis")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.grid(True)

plt.show()

# ==========================================
# Explained Variance
# ==========================================

plt.figure(figsize=(7,5))

plt.bar(
    ["PC1","PC2"],
    pca.explained_variance_ratio_
)

plt.title("Explained Variance Ratio")

plt.ylabel("Variance")

plt.show()

# ==========================================
# Cumulative Explained Variance
# ==========================================

plt.figure(figsize=(7,5))

plt.plot(
    range(1,3),
    pca.explained_variance_ratio_.cumsum(),
    marker='o',
    linewidth=2
)

plt.xticks([1,2])

plt.xlabel("Number of Principal Components")

plt.ylabel("Cumulative Explained Variance")

plt.title("Cumulative Explained Variance")

plt.grid(True)

plt.show()