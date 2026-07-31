# 🧠 Machine Learning Clustering Algorithms

This repository contains implementations of popular **Unsupervised Machine Learning** algorithms using the **Mall Customers Dataset**. Each algorithm is implemented from scratch using Python and Scikit-learn, along with visualizations, result datasets, and documentation.

---

# 📌 Algorithms Included

## ✅ 1. K-Means Clustering

**Description**

K-Means is a partition-based clustering algorithm that groups similar data points into **K clusters** based on the nearest centroid.

### Features

- Elbow Method
- K-Means++ Initialization
- Cluster Visualization
- Cluster Distribution
- Result CSV

---

## ✅ 2. Hierarchical Clustering

**Description**

Hierarchical Clustering builds a hierarchy of clusters using a bottom-up (Agglomerative) approach and visualizes relationships through a **Dendrogram**.

### Features

- Dendrogram
- Agglomerative Clustering
- Cluster Visualization
- Result CSV

---

## ✅ 3. DBSCAN (Density-Based Spatial Clustering)

**Description**

DBSCAN groups points based on density and automatically identifies **noise (outliers)** without requiring the number of clusters beforehand.

### Features

- Density-Based Clustering
- Noise Detection
- Cluster Visualization
- Result CSV

---

## ✅ 4. Gaussian Mixture Model (GMM)

**Description**

Gaussian Mixture Model is a probabilistic clustering algorithm that assumes data is generated from a mixture of Gaussian distributions.

### Features

- Soft Clustering
- Probability-Based Clustering
- Cluster Visualization
- Result CSV

---

## ✅ 5. Principal Component Analysis (PCA)

**Description**

PCA is a **Dimensionality Reduction** technique used to reduce the number of features while preserving maximum information.

### Features

- Feature Scaling
- Principal Components
- Explained Variance
- PCA Scatter Plot
- Reduced Dataset CSV

---

# 📂 Dataset

**Mall_Customers.csv**

Features:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

---

# 📁 Project Structure

```text
ML_Clustering_Algorithms
│
├── data
│   ├── Mall_Customers.csv
│   ├── Mall_Customers_All_Clusters.csv
│   └── Mall_Customers_PCA.csv
│
├── KMeans_Clustering
│
├── Hierarchical_Clustering
│
├── DBSCAN_Clustering
│
├── Gaussian_Mixture_Model
│
├── PCA_Dimensionality_Reduction
│
├── merge_cluster_results_process.py
│
├── README.md
│
└── requirements.txt
```

---

# 📊 Visualizations Included

## K-Means

- Elbow Method
- Cluster Visualization
- Cluster Distribution

---

## Hierarchical

- Dendrogram
- Cluster Visualization

---

## DBSCAN

- Cluster Visualization
- Noise Detection

---

## Gaussian Mixture Model

- Cluster Visualization
- Cluster Distribution

---

## PCA

- PCA Scatter Plot
- Explained Variance
- Cumulative Explained Variance

---

# 📈 Comparison of Algorithms

| Algorithm | Type | Need K | Detect Noise | Soft Clustering |
|------------|------|--------|--------------|-----------------|
| K-Means | Partition Based | ✅ | ❌ | ❌ |
| Hierarchical | Tree Based | ✅ | ❌ | ❌ |
| DBSCAN | Density Based | ❌ | ✅ | ❌ |
| Gaussian Mixture Model | Probabilistic | ✅ | ❌ | ✅ |
| PCA | Dimensionality Reduction | ❌ | ❌ | ❌ |

---

# 📄 Output Files

The project generates the following output files:

- Mall_Customers_KMeans.csv
- Mall_Customers_Hierarchical.csv
- Mall_Customers_DBSCAN.csv
- Mall_Customers_GMM.csv
- Mall_Customers_PCA.csv
- Mall_Customers_All_Clusters.csv

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Machine-learning.git
```

Go to the project

```bash
cd Machine-learning/ML_Clustering_Algorithms
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Run any algorithm individually.

Example:

```bash
python KMeans_Clustering/kmeans_clustering.py
```

```bash
python Hierarchical_Clustering/hierarchical_clustering.py
```

```bash
python DBSCAN_Clustering/dbscan_clustering.py
```

```bash
python Gaussian_Mixture_Model/Gaussian_Mixture_Model.py
```

```bash
python PCA_Dimensionality_Reduction/PCA.py
```

To merge clustering results:

```bash
python merge_cluster_results_process.py
```

---

# 📌 Learning Outcomes

This project demonstrates:

- Unsupervised Machine Learning
- Partition-Based Clustering
- Hierarchical Clustering
- Density-Based Clustering
- Probabilistic Clustering
- Dimensionality Reduction
- Data Visualization
- Customer Segmentation
- Feature Engineering
- Cluster Analysis

---

# 👨‍💻 Author

**Ajim Mujawar**

- 💻 Python Developer
- 📊 Data Science & Machine Learning Enthusiast
- 🤖 AI & Machine Learning Learner

---

⭐ If you found this repository helpful, consider giving it a **Star** on GitHub!