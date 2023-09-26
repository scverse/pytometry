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
