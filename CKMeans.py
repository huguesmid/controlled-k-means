# -*- coding: utf-8 -*-
"""
CKMeans: Controlled K-Means clustering | Hybrid-Controlled K-Means clustering

This module provides an implementation of controlled and hybrid-controlled
K-Means algorithms.

Author: Midingoyi Mahuton Hugues
2026, March

References:
- scikit-learn KMeans implementation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- K-means with Controlled Center Updates: Rethinking the Centroid Update Rule (pre-print)
"""

import numpy as np

class CKMeans:
    """
    Controlled K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.

    init : {'k-means++', 'random'} or ndarray of shape (n_clusters, n_features), default='k-means++'

    n_init : int, default=10
        Number of runs.

    init_random_seed : int or None, default=None

    max_iter : int or None, default=None
        If None, runs until convergence.

    verbose : bool, default=False

    tol : float, default=1e-4

    algorithm : {'controlled', 'hybrid-controlled'}, default='controlled'

    alpha_interval : tuple(float, float), default=(1.0, 1.9)

    alpha_random_seed : int or None, default=None

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)

    labels_ : ndarray of shape (n_points,)

    inertia_ : float

    n_iter_ : list

    alpha_values_ : list
    """

    def __init__(
        self,
        n_clusters,
        init="k-means++",
        n_init=10,
        init_random_seed=None,
        max_iter=None,
        verbose=False,
        tol=1e-4,
        algorithm="controlled",
        alpha_interval=(1.0, 1.9),
        alpha_random_seed=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.init_random_seed = init_random_seed
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.algorithm = algorithm
        self.alpha_interval = alpha_interval
        self.alpha_random_seed = alpha_random_seed

        self._validate_params()

    # =========================
    # Validation
    # =========================
    def _validate_params(self):
        if not isinstance(self.n_clusters, int) or self.n_clusters <= 0:
            raise ValueError("n_clusters must be > 0.")

        if isinstance(self.init, str):
            if self.init not in {"k-means++", "random"}:
                raise ValueError("Invalid init.")
        else:
            arr = np.asarray(self.init)
            if arr.shape[0] != self.n_clusters:
                raise ValueError("Invalid init shape.")

        if self.max_iter is not None:
            if not isinstance(self.max_iter, int) or self.max_iter <= 0:
                raise ValueError("max_iter must be positive or None.")

        a_min, a_max = self.alpha_interval
        if not (0 < a_min <= a_max < 2):
            raise ValueError("alpha must be in (0,2).")

    # =========================
    # Fit
    # =========================
    def fit(self, X):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")

        n_points = X.shape[0]

        if self.n_clusters > n_points:
            raise ValueError("Too many clusters.")

        rng_init = np.random.default_rng(self.init_random_seed)
        rng_alpha = np.random.default_rng(self.alpha_random_seed)

        best_inertia = None

        self.n_iter_ = []
        self.alpha_values_ = []

        # Choose algorithm
        if self.algorithm == "controlled":
            algo = self._ckmeans
        else:
            algo = self._hckmeans

        if self.verbose:
            print(f"CKMeans fitting started with {self.n_init} runs")

        for run in range(self.n_init):

            if self.verbose:
                print(f"\n--- Run {run + 1}/{self.n_init} ---")

            # alpha
            a_min, a_max = self.alpha_interval
            alpha = a_min if a_min == a_max else rng_alpha.uniform(a_min, a_max)
            self.alpha_values_.append(alpha)

            if self.verbose:
                print(f"alpha = {alpha:.4f}")

            # init
            if isinstance(self.init, str):
                if self.init == "random":
                    idx = rng_init.choice(n_points, self.n_clusters, replace=False)
                    centers = X[idx]
                else:
                    centers = self._init_kmeans_pp(X, rng_init)
            else:
                centers = np.asarray(self.init).copy()

            if self.verbose:
                print(f"Initialization: {self.init}")

            # run algo
            centers, labels, n_iter = algo(X, centers, alpha)

            if self.verbose:
                print(f"Iterations: {n_iter}")

            inertia = self._compute_inertia(X, centers, labels)

            if self.verbose:
                print(f"Inertia: {inertia:.4f}")

            self.n_iter_.append(n_iter)

            if best_inertia is None or inertia < best_inertia:
                if self.verbose:
                    print("→ New best solution found")

                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = inertia

        if self.verbose:
            print("\nCKMeans fitting completed")
            print(f"Best inertia: {self.inertia_:.4f}")

        return self
    # ========================
    # PREDICT
    # =========================
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """

        # Check that the model has been fitted
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("This CKMeans instance is not fitted yet. Call 'fit' first.")

        X = np.asarray(X)

        # Ensure X is a 2D array
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        # Check feature dimension consistency
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                "Number of features in X does not match training data."
            )

        # Assign each sample to the nearest cluster center
        labels = np.argmin(
            np.sum((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2),
            axis=1,
        )

        return labels

    # =========================
    # Initialization
    # =========================
    def _init_kmeans_pp(self, X, rng):
        n = X.shape[0]
        centers = []

        idx = rng.integers(n)
        centers.append(X[idx])

        for _ in range(1, self.n_clusters):
            dist_sq = np.min(
                np.sum((X[:, None] - np.array(centers)) ** 2, axis=2),
                axis=1,
            )
            probs = dist_sq / dist_sq.sum()
            idx = rng.choice(n, p=probs)
            centers.append(X[idx])

        return np.array(centers)

    # =========================
    # Core methods
    # =========================
    def _assign(self, X, centers):
        return np.argmin(
            np.sum((X[:, None] - centers[None, :]) ** 2, axis=2), axis=1
        )

    def _compute_inertia(self, X, centers, labels):
        diff = X - centers[labels]
        return np.sum(diff ** 2)

    # =========================
    # Controlled k-means
    # =========================
    def _ckmeans(self, X, centers, alpha):
        k = centers.shape[0]

        C = centers.copy()
        Xc = centers.copy()
        I = np.zeros(k, dtype=int)

        labels = self._assign(X, Xc)
        it = 0

        while True:
            it += 1
            old_Xc = Xc.copy()

            for j in range(k):
                if I[j] == 0:
                    pts = X[labels == j]

                    if len(pts) > 0:
                        C[j] = pts.mean(axis=0)

                    Xc[j] = old_Xc[j] + alpha * (C[j] - old_Xc[j])

                    if np.linalg.norm(C[j] - Xc[j]) < self.tol:
                        I[j] = 1
                        Xc[j] = C[j]

            new_labels = self._assign(X, Xc)

            if np.all(new_labels == labels) and np.all(I == 1):
                break

            labels = new_labels

            if self.max_iter is not None and it >= self.max_iter:
                if self.verbose:
                    print("Max iter reached")
                break

        return Xc, labels, it

    # =========================
    # Hybrid Controlled k-means
    # =========================
    def _hckmeans(self, X, centers, alpha):
        k = centers.shape[0]

        C = centers.copy()
        Xc = centers.copy()
        I = np.zeros(k, dtype=int)

        labels = self._assign(X, Xc)
        it = 0

        while True:
            it += 1
            old_Xc = Xc.copy()

            for j in range(k):
                if I[j] == 0:
                    pts = X[labels == j]

                    if len(pts) > 0:
                        C[j] = pts.mean(axis=0)

                    Xc[j] = old_Xc[j] + alpha * (C[j] - old_Xc[j])

                    if np.linalg.norm(C[j] - Xc[j]) < self.tol:
                        I[j] = 1
                        Xc[j] = C[j]

            new_labels = self._assign(X, Xc)

            if np.all(new_labels == labels):
                Xc = C.copy()
                new_labels = self._assign(X, Xc)

                if np.all(new_labels == labels):
                    break

            labels = new_labels

            if self.max_iter is not None and it >= self.max_iter:
                if self.verbose:
                    print("Max iter reached")
                break

        return Xc, labels, it