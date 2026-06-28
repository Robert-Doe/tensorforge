"""module_13/kmeans.py — k-Means Clustering from scratch.

Implements the standard Lloyd's algorithm:
  1. Initialise k centroids (random sample from data)
  2. Assign each point to its nearest centroid
  3. Move each centroid to the mean of its assigned points
  4. Repeat 2-3 until assignments stop changing (convergence)
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
MAX_ITERATIONS  = 300    # hard cap — prevents infinite loops on pathological data
TOLERANCE       = 1e-4   # centroid shift below this means convergence


def euclidean_distances_to_centroids(X, centroids):
    """Compute distance from every point to every centroid.

    Uses broadcasting to avoid an explicit Python loop over samples.

    Args:
        X:         (n_samples, n_features) — data points
        centroids: (k, n_features)         — current centroid positions

    Returns:
        distances: (n_samples, k) — dist[i, j] = distance from point i to centroid j
    """
    # X[:, np.newaxis] → (n, 1, d); centroids → (k, d); diff → (n, k, d)
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))   # → (n_samples, k)


def assign_clusters(X, centroids):
    """Return the index of the nearest centroid for each point.

    Args:
        X:         (n_samples, n_features)
        centroids: (k, n_features)

    Returns:
        labels: (n_samples,) int array of cluster indices in [0, k)
    """
    distances = euclidean_distances_to_centroids(X, centroids)
    return np.argmin(distances, axis=1)   # index of closest centroid per row


def update_centroids(X, labels, k):
    """Move each centroid to the mean of its assigned points.

    If a cluster becomes empty (edge case), reinitialise its centroid
    to a random point — prevents NaN centroids.

    Args:
        X:      (n_samples, n_features)
        labels: (n_samples,) cluster assignments from assign_clusters
        k:      number of clusters

    Returns:
        new_centroids: (k, n_features)
    """
    n_features   = X.shape[1]
    new_centroids = np.zeros((k, n_features))
    rng           = np.random.default_rng(RANDOM_SEED)

    for cluster_idx in range(k):
        members = X[labels == cluster_idx]
        if len(members) == 0:
            # empty cluster: reinitialise to a random point (rare but possible)
            new_centroids[cluster_idx] = X[rng.integers(len(X))]
        else:
            new_centroids[cluster_idx] = members.mean(axis=0)

    return new_centroids


def inertia(X, labels, centroids):
    """Sum of squared distances from each point to its assigned centroid.

    Lower inertia = tighter, more compact clusters.
    Used to compare k values (elbow method).

    Args:
        X:         (n_samples, n_features)
        labels:    (n_samples,) cluster assignments
        centroids: (k, n_features)

    Returns:
        float: total inertia (sum of squared distances)
    """
    total = 0.0
    for i, x in enumerate(X):
        diff   = x - centroids[labels[i]]
        total += float(diff @ diff)    # squared Euclidean distance
    return total


class KMeans:
    """k-Means clustering using Lloyd's algorithm.

    Attributes:
        k:           number of clusters
        centroids_:  (k, n_features) final centroid positions after fit
        labels_:     (n_samples,) cluster assignment for training data
        inertia_:    float — total within-cluster sum of squares
        n_iter_:     int   — number of iterations until convergence
        history_:    list of inertia values per iteration (for plotting)
    """

    def __init__(self, k=3):
        """Initialise KMeans.

        Args:
            k: number of clusters to form
        """
        self.k        = k
        self.centroids_ = None
        self.labels_    = None
        self.inertia_   = None
        self.n_iter_    = 0
        self.history_   = []

    def fit(self, X):
        """Run Lloyd's algorithm on X until convergence or MAX_ITERATIONS.

        Initialisation: pick k distinct random rows from X as starting centroids
        (k-means++ would be better — see DECISIONS.md for why we don't use it here).

        Args:
            X: (n_samples, n_features) array of unlabelled data points

        Returns:
            self (allows chaining: km.fit(X).labels_)
        """
        rng = np.random.default_rng(RANDOM_SEED)

        # randomly pick k distinct points as initial centroids
        init_idx       = rng.choice(len(X), size=self.k, replace=False)
        self.centroids_ = X[init_idx].copy()   # (k, n_features)

        labels = assign_clusters(X, self.centroids_)
        self.history_.append(inertia(X, labels, self.centroids_))

        for iteration in range(MAX_ITERATIONS):
            new_centroids = update_centroids(X, labels, self.k)
            new_labels    = assign_clusters(X, new_centroids)

            # check convergence: did centroids move less than TOLERANCE?
            centroid_shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            labels          = new_labels
            self.history_.append(inertia(X, labels, self.centroids_))

            if centroid_shift < TOLERANCE:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = MAX_ITERATIONS   # didn't converge — hit the cap

        self.labels_   = labels
        self.inertia_  = self.history_[-1]
        return self

    def predict(self, X):
        """Assign new points to the nearest trained centroid.

        Args:
            X: (n_samples, n_features)

        Returns:
            labels: (n_samples,) cluster indices
        """
        assert self.centroids_ is not None, "Call fit() before predict()"
        return assign_clusters(X, self.centroids_)


def elbow_sweep(X, k_range):
    """Fit KMeans for each k and return inertia values.

    The "elbow" — where adding another cluster gives diminishing returns —
    is the natural choice of k.

    Args:
        X:       (n_samples, n_features)
        k_range: iterable of k values to try

    Returns:
        inertias: list of float, one per k value
    """
    inertias = []
    for k in k_range:
        km = KMeans(k=k)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"  k={k:2d}  inertia={km.inertia_:10.2f}  iters={km.n_iter_}")
    return inertias
