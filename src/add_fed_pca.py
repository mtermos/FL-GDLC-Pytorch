import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute root-mean-square error between two arrays.
    """
    return float(np.sqrt(mean_squared_error(actual, predicted)))


@dataclass
class FederatedPCAClient:
    """
    Encapsulates local and global PCA operations for a single client.
    """
    name: str
    df: pd.DataFrame
    n_components: int = 2
    local_scores: np.ndarray = field(init=False)
    scaler_local: StandardScaler = field(init=False)
    global_scores: np.ndarray = field(init=False)
    scaler_global: StandardScaler = field(init=False)

    def compute_local_pca(self) -> np.ndarray:
        """
        Perform local PCA.
        Returns the covariance matrix of the local PCA scores.
        """

        self.df.fillna(0, inplace=True)
        self.scaler_local = StandardScaler()
        X_std = self.scaler_local.fit_transform(self.df)
        pca = PCA(n_components=self.n_components)
        self.local_scores = pca.fit_transform(X_std)

        # Return covariance of local PCA scores
        return np.cov(self.local_scores, rowvar=False)

    def apply_global_pca(self, global_components: np.ndarray) -> np.ndarray:
        """
        Project local PCA scores into the global PCA subspace and standardize.
        """
        transformed = self.local_scores.dot(global_components)
        self.scaler_global = StandardScaler().fit(transformed)
        self.global_scores = self.scaler_global.transform(transformed)
        return self.global_scores


def process_clients_with_grouped_pca_rmse(
        dfs_dict: Dict[str, pd.DataFrame],
        n_components: int = 2
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]], List[str]]:
    """
    Perform two-stage federated PCA with RMSE-based reconstruction errors.

    Returns:
        client_dfs: final DataFrames per client with 'global_pca_*' columns.
        errors: dictionary with 'reconstruction_errors_local' and 'reconstruction_errors_federated'.
        pca_columns: list of names for global PCA columns.
    """

    # Local stage: compute per-client covariance
    clients = []
    cov_matrices = []

    for name, df in dfs_dict.items():
        logger.info("Computing local PCA for client %s", name)
        client = FederatedPCAClient(
            name=name,
            df=df,
            n_components=n_components
        )
        try:
            cov = client.compute_local_pca()
        except ValueError as e:
            logger.warning(str(e))
            continue
        clients.append(client)
        cov_matrices.append(cov)

    if not cov_matrices:
        raise RuntimeError("No valid clients for PCA.")

    # Global stage: aggregate and compute shared components
    global_cov = np.mean(cov_matrices, axis=0)
    eigvals, eigvecs = np.linalg.eigh(global_cov)
    top_indices = np.argsort(eigvals)[::-1][:n_components]
    global_components = eigvecs[:, top_indices]

    # Prepare outputs
    reconstruction_errors_local = {}
    reconstruction_errors_federated = {}
    pca_columns = [f"global_pca_{i+1}" for i in range(n_components)]

    # Apply global PCA, compute RMSE, and save results
    for client in clients:
        logger.info("Applying global PCA for client %s", client.name)
        G = client.apply_global_pca(global_components)

        # Build final DataFrame
        client.df = pd.DataFrame(G, columns=pca_columns, index=client.df.index)

        # Compute RMSE metrics
        Z_std = client.df.dot(global_components)
        Z_std = client.scaler_global.transform(Z_std)

        rmse_loc = compute_rmse(
            client.scaler_global.transform(client.local_scores),
            client.scaler_global.transform(
                client.local_scores.dot(global_components.T))
        )
        rmse_fed = compute_rmse(
            client.scaler_global.transform(client.local_scores), G
        )
        reconstruction_errors_local[client.name] = rmse_loc
        reconstruction_errors_federated[client.name] = rmse_fed

        logger.info(
            "Client %s errors: local RMSE=%.4f, federated RMSE=%.4f",
            client.name, rmse_loc, rmse_fed
        )

    errors = {
        'reconstruction_errors_local': reconstruction_errors_local,
        'reconstruction_errors_federated': reconstruction_errors_federated
    }

    return {client.name: client.df for client in clients}, errors, pca_columns
