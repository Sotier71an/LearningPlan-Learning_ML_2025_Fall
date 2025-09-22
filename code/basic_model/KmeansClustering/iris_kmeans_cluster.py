from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

import os

# generate a simple 2-Dimension dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# training K-means model
model = KMeans(n_clusters=4)
model.fit(X)

# predict cluster labels
y_kmeans = model.predict(X)

# visualize clustering results
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Add cluster centers
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')

# Add title and labels
plt.title('K-means Clustering Results', fontsize=16)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# save picture BEFORE showing
output_dir = 'code/basic_model/KmeansClustering/plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'iris_kmeans_results.png'),
            dpi=300, bbox_inches='tight')
print(f"Plot saved to {os.path.join(output_dir, 'iris_kmeans_results.png')}")

# show picture AFTER saving
plt.show()