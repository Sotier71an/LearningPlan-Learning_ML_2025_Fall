from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import os

# loading Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Reduce dimensions to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# visualize results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA of Iris Dataset')

output_dir = 'code/basic_model/PCA/plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'iris_pca_results.png'),
            dpi=300, bbox_inches='tight')
print(f"Plot saved to {os.path.join(output_dir, 'iris_pca_results.png')}")

plt.show()