# ==========================================
# K-Means Clustering
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans

# Import Dataset
dataset = pd.read_csv(r"E:\ML\Machine learning\ML_Clustering_Algorithms\data\Mall_Customers.csv")

# Selecting Annual Income and Spending Score
X = dataset.iloc[:, [3, 4]].values

# ==========================================
# Elbow Method
# ==========================================

wcss = []

for i in range(1, 11):

    model = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    model.fit(X)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(1,11),wcss,marker='o')

plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.grid(True)

plt.show()

# ==========================================
# Train K-Means
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# ==========================================
# Save Output
# ==========================================

dataset["Cluster"] = y_kmeans

dataset.to_csv("Mall_Customers_KMeans.csv",index=False)

print(os.getcwd())

print("CSV File Saved Successfully")

# ==========================================
# Visualization
# ==========================================

plt.figure(figsize=(10,7))

for i in range(5):

    plt.scatter(
        X[y_kmeans==i,0],
        X[y_kmeans==i,1],
        s=80,
        label=f"Cluster {i+1}"
    )

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=250,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title("K-Means Clustering")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.legend()

plt.grid(True)

plt.show()