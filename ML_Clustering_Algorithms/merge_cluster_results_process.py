import pandas as pd

# Read merged clustering file
clusters = pd.read_csv("Mall_Customers_All_Clusters.csv")

# Read PCA file
pca = pd.read_csv("Mall_Customers_PCA.csv")

# Add PCA columns
clusters["PC1"] = pca["PC1"]
clusters["PC2"] = pca["PC2"]

# Save updated file
clusters.to_csv(
    "Mall_Customers_All_Clusters.csv",
    index=False
)

print("✅ PCA merged successfully!")
print(clusters.head())