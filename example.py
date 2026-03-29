import numpy as np
from CKMeans import CKMeans

# -----------------------------
# Generate data (3 clusters)
# -----------------------------
np.random.seed(0)

cluster1 = np.random.randn(100, 2) + np.array([0, 0])
cluster2 = np.random.randn(100, 2) + np.array([5, 5])
cluster3 = np.random.randn(100, 2) + np.array([0, 5])

X = np.vstack([cluster1, cluster2, cluster3])

# -----------------------------
# Apply CKMeans
# -----------------------------
model = CKMeans(
    n_clusters=3,
    init="k-means++",
    n_init=5,
    init_random_seed=42,
    max_iter=100,
    alpha_interval=(1.0, 1.9),
    alpha_random_seed=41,
    algorithm="controlled",
    verbose=False,
)

model.fit(X)

# -----------------------------
# Results
# -----------------------------
print("Inertia:", model.inertia_)
print("Centers:\n", model.cluster_centers_)
print("Iterations per run:", model.n_iter_)
print("Alpha values:", model.alpha_values_)

# -----------------------------
# Prediction
# -----------------------------
labels = model.predict(X)

