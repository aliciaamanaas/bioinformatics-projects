import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load dataset
# -----------------------------
data = load_breast_cancer()

# Features (cell measurements) and labels (0 = malignant, 1 = benign)
X = data.data
y = data.target

# Create DataFrame for easier handling
df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y

print(df.head())

# -----------------------------
# Basic exploration
# -----------------------------
# Check class distribution
print(df["target"].value_counts())

# -----------------------------
# Correlation heatmap
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------
# PCA (Dimensionality reduction)
# -----------------------------
X_features = df.drop("target", axis=1)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_features)

# Plot PCA results
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["target"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Breast Cancer Dataset")
plt.show()

# -----------------------------
# Machine Learning Model
# -----------------------------
# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, preds)
print(f"Model accuracy: {accuracy:.3f}")