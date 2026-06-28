"""module_14/pca.py — Principal Component Analysis from scratch.

PCA steps:
  1. Centre the data (subtract mean per feature)
  2. Compute the covariance matrix  C = (X_centred.T @ X_centred) / (n-1)
  3. Eigendecompose C → eigenvectors (principal components) + eigenvalues
  4. Sort by eigenvalue descending (most variance first)
  5. Project data onto the top-k eigenvectors
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_N_COMPONENTS = 2   # default projection dimensionality


class PCA:
    """Principal Component Analysis using eigendecomposition of the covariance matrix.

    Attributes:
        n_components:        int — number of principal components to keep
        components_:         (n_components, n_features) — the eigenvectors (PCs)
        explained_variance_: (n_components,) — eigenvalue for each PC
        explained_variance_ratio_: (n_components,) — fraction of total variance per PC
        mean_:               (n_features,) — per-feature mean subtracted during fit
    """

    def __init__(self, n_components=DEFAULT_N_COMPONENTS):
        """Initialise PCA.

        Args:
            n_components: number of dimensions to keep after projection
        """
        self.n_components              = n_components
        self.components_               = None
        self.explained_variance_       = None
        self.explained_variance_ratio_ = None
        self.mean_                     = None

    def fit(self, X):
        """Learn principal components from X.

        Args:
            X: (n_samples, n_features) array — will NOT be modified

        Returns:
            self (for chaining)
        """
        n_samples, n_features = X.shape

        # Step 1: centre — each column now has mean zero
        self.mean_   = X.mean(axis=0)
        X_centred    = X - self.mean_

        # Step 2: covariance matrix (n_features × n_features)
        # Dividing by (n-1) gives the unbiased sample covariance
        cov_matrix = (X_centred.T @ X_centred) / (n_samples - 1)

        # Step 3: eigendecomposition — numpy returns unsorted eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # eigh (not eig) is faster and more stable for real symmetric matrices

        # Step 4: sort descending by eigenvalue (largest variance first)
        order        = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]   # columns are eigenvectors

        # Keep only the top n_components
        self.components_         = eigenvectors[:, :self.n_components].T
        # .T so shape is (n_components, n_features) — each row is one PC

        self.explained_variance_ = eigenvalues[:self.n_components]
        total_variance           = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X):
        """Project X onto the principal components.

        Args:
            X: (n_samples, n_features) — same feature space as fit()

        Returns:
            X_projected: (n_samples, n_components)
        """
        assert self.components_ is not None, "Call fit() before transform()"
        X_centred = X - self.mean_                  # use mean from fit
        return X_centred @ self.components_.T        # (n, d) @ (d, k) → (n, k)

    def fit_transform(self, X):
        """Fit and transform in one call.

        Args:
            X: (n_samples, n_features)

        Returns:
            X_projected: (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def print_summary(self, feature_names=None):
        """Print explained variance table and component loadings.

        Args:
            feature_names: optional list of feature name strings
        """
        assert self.components_ is not None, "Call fit() first"

        print("\nExplained Variance per Component:")
        cumulative = 0.0
        for i, (var, ratio) in enumerate(
            zip(self.explained_variance_, self.explained_variance_ratio_)
        ):
            cumulative += ratio
            bar = "█" * int(ratio * 40)
            print(f"  PC{i+1}: {ratio*100:5.1f}%  cumulative={cumulative*100:5.1f}%  {bar}")

        if feature_names is not None:
            print("\nComponent Loadings (how much each feature contributes):")
            print(f"  {'Feature':<22} " +
                  "  ".join(f"{'PC'+str(i+1):>7}" for i in range(self.n_components)))
            print("  " + "-" * (22 + 9 * self.n_components))
            for fi, fname in enumerate(feature_names):
                loadings = "  ".join(
                    f"{self.components_[ci, fi]:>7.3f}"
                    for ci in range(self.n_components)
                )
                print(f"  {fname:<22} {loadings}")
