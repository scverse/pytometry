from __future__ import annotations

import logging

import numpy as np
from anndata import AnnData
from consensusclustering import ConsensusClustering
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm

logger = logging.getLogger("pytometry.flowsom")


def som_clustering(
    x: np.ndarray,
    som_dim: tuple[int, int] = (10, 10),
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    batch_size: int = 500,
    seed: int = 42,
    weight_init: str = "random",
    neighbourhood_function: str = "gaussian",
    verbose: bool = False,
) -> MiniSom:
    """Train a SOM on the given cytometry data.

    Args:
        x (numpy.array): cytometry data of shape (n_samples, n_features)
        som_dim (tuple[int, int], default=(10, 10)): dimensions of the SOM
        sigma (float, default=1.0): radius of the different neighbourhoods in the SOM
        learning_rate (float, default=0.5): learning rate of the SOM
        batch_size (int, default=500): batch size for training the SOM
        seed (int, default=42): random seed for reproducibility
        weight_init (str, default="random"): weight initialization method, either
        "random" or "pca". NOTE: if "pca" is chosen, the input data must be scaled to
        zero mean and unit variance.
        neighbourhood_function (str, default="gaussian"): neighbourhood function for
        the SOM
        verbose (bool, default=False)

    Returns:
        MiniSom: trained SOM
    """
    som = MiniSom(
        x=som_dim[0],
        y=som_dim[1],
        input_len=x.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=seed,
        neighborhood_function=neighbourhood_function,
    )
    if weight_init == "random":
        som.random_weights_init(x)
    elif weight_init == "pca":
        som.pca_weights_init(x)
    else:
        raise ValueError('Unknown weight_init, must be one of "random" or "pca"')
    logger.info("Training SOM")
    som.train_batch(x, batch_size, verbose=verbose)
    return som


def meta_clustering(
    som: MiniSom,
    n_features: int,
    min_clusters: int = 2,
    max_clusters: int = 10,
    n_resamples: int = 100,
    resample_frac: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    """Meta-clustering of SOM nodes using consensus clustering.

    Consensus clustering is implemented using the `consensusclustering` package (see
    https://github.com/burtonrj/consensusclustering for details).

    Args:
        som (MiniSom): trained SOM
        n_features (int): number of features in the cytometry data
        min_clusters (int, default=2): minimum number of clusters to consider
        max_clusters (int, default=10): maximum number of clusters to consider
        n_resamples (int, default=100): number of resamples for consensus clustering
        resample_frac (float, default=0.5): fraction of samples to resample for
        verbose (bool, default=False)

    Returns:
        numpy.array: meta-clustering of SOM nodes
    """
    clustering_obj = AgglomerativeClustering(affinity="euclidean", linkage="ward")
    cc = ConsensusClustering(
        clustering_obj,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        n_resamples=n_resamples,
        resample_frac=resample_frac,
    )
    weights = som.get_weights()
    flatten_weights = weights.reshape(
        som._activation_map.shape[0] * som._activation_map.shape[1], n_features
    )
    cc.fit(flatten_weights, progress_bar=verbose)
    k = cc.best_k()
    clustering_obj.set_params(**{"n_clusters": k})
    meta_class = clustering_obj.fit_predict(flatten_weights)
    return meta_class.reshape(
        som._activation_map.shape[0], som._activation_map.shape[1]
    )


def flowsom_clustering(
    adata: AnnData,
    key_added: str = "clusters",
    som_dim: tuple[int, int] = (10, 10),
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    batch_size: int = 500,
    seed: int = 42,
    weight_init: str = "random",
    neighbourhood_function: str = "gaussian",
    min_clusters: int = 2,
    max_clusters: int = 10,
    n_resamples: int = 100,
    resample_frac: float = 0.5,
    copy: bool = False,
    verbose: bool = False,
) -> AnnData:
    """Cluster cytometry data using FlowSOM.

    Based on the original FlowSOM R package by Van Gassen et al. (2015)
    (https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.22625.) and the python
    implementation found at https://github.com/Hatchin/FlowSOM.

    Args:
        adata (AnnData): annotated data matrix of shape (n_samples, n_features)
        key_added (str, default="clusters"): key under which to add the cluster labels
        som_dim (tuple[int, int], default=(10, 10)): dimensions of the SOM
        sigma (float, default=1.0): radius of the different neighbourhoods in the SOM
        learning_rate (float, default=0.5): learning rate of the SOM
        batch_size (int, default=500): batch size for training the SOM
        seed (int, default=42): random seed for reproducibility
        weight_init (str, default="random"): weight initialization method, either
        "random" or "pca". NOTE: if "pca" is chosen, the input data must be scaled to
        zero mean and unit variance.
        neighbourhood_function (str, default="gaussian"): neighbourhood function for
        the SOM
        min_clusters (int, default=2): minimum number of clusters to consider
        max_clusters (int, default=10): maximum number of clusters to consider
        n_resamples (int, default=100): number of resamples for consensus clustering
        resample_frac (float, default=0.5): fraction of samples to resample for
        copy (bool, default=False): whether to copy the AnnData object or modify it
        verbose (bool, default=False)

    Returns:
        AnnData: annotated data matrix with cluster labels added under `key_added`

    """
    adata = adata.copy() if copy else adata
    logger.info("Running FlowSOM clustering")
    logger.info("Training SOM")
    som = som_clustering(
        adata.X,
        som_dim=som_dim,
        sigma=sigma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        weight_init=weight_init,
        neighbourhood_function=neighbourhood_function,
        verbose=verbose,
    )
    logger.info("Meta-clustering of SOM nodes")
    meta_class = meta_clustering(
        som,
        adata.X.shape[1],
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        n_resamples=n_resamples,
        resample_frac=resample_frac,
    )
    logger.info("Assigning cluster labels to cells")
    labels = []
    for i in tqdm(
        range(adata.shape[0]),
        desc="Assigning cluster labels to cells",
        total=adata.shape[0],
        disable=not verbose,
    ):
        xx = adata.X[i, :]
        winner = som.winner(xx)
        labels.append(meta_class[winner])
    adata.obs[key_added] = labels
    return adata
