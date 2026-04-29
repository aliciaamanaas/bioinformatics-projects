import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -----------------------------
# 1. Load GEO expression dataset
# -----------------------------
df = pd.read_csv(
    "data/GSE45827_series_matrix.txt",
    sep="\t",
    comment="!",
    low_memory=False
)

# -----------------------------
# 2. Prepare expression matrix
# -----------------------------
# Remove gene identifier column
X = df.drop(columns=["ID_REF"])

# Convert all values to numeric (handles strings -> NaN if needed)
X = X.apply(pd.to_numeric, errors="coerce")

# Transpose so that:
# rows = samples
# columns = genes
X = X.T

# Handle missing values (replace with column mean) to avoid losing too much biological information
X = X.fillna(X.mean())

# -----------------------------
# 3. Normalize data
# -----------------------------
# Standardization is essential for gene expression analysis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. Dimensionality reduction (PCA)
# -----------------------------
# Reduce high-dimensional gene space to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize PCA results
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA - Breast Cancer Gene Expression")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -----------------------------
# 5. Unsupervised clustering
# -----------------------------
# KMeans attempts to group samples based on gene expression similarity
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Visualize clustering results in PCA space
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.title("KMeans Clustering of Samples")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()