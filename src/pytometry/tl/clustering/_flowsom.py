from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypedDict

import graspologic_native as gn
import numpy as np
import pandas as pd
from anndata import AnnData
from consensusclustering import ConsensusClustering
from natsort import natsorted
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from pytometry._types import RNGLike, SeedLike

if TYPE_CHECKING:
    from fastersom import SOM, KMeans

logger = logging.getLogger("pytometry.flowsom")

# pytometry's names of the initialisations, as fastersom calls them
_WEIGHT_INITS = {"random": "sample", "pca": "pca"}


def som_clustering(
    x: np.ndarray,
    som_dim: tuple[int, int] = (10, 10),
    sigma: float | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    seed: int = 42,
    weight_init: Literal["random", "pca"] = "random",
    neighbourhood_function: Literal["gaussian"] = "gaussian",
    verbose: bool = False,
    n_epochs: int = 10,
    metric: Literal["euclidean", "manhattan", "cosine", "correlation"] = "euclidean",
    device: str = "cpu",
    som_kwargs: dict | None = None,
) -> SOM:
    """Train a batch SOM on the given cytometry data with :class:`fastersom.SOM`.

    Parameters
    ----------
    x
        cytometry data of shape `(n_samples, n_features)`
    som_dim
        dimensions of the SOM
    sigma
        initial width of the gaussian neighbourhood in grid units, which decays to a tenth of it (at most one grid
        unit). If None, defaults to a quarter of the shorter side of the SOM (rounded up to half a unit).
    learning_rate
        deprecated, has no effect: the batch SOM moves each node to the neighbourhood-weighted mean of the cells.
    batch_size
        number of cells processed at once. Affects memory and speed only, not the result. If None, chosen by fastersom.
    seed
        random seed for reproducibility
    weight_init
        weight initialization method: "random" starts from randomly chosen cells, "pca" from a regular grid in the
        plane of the first two principal components of the data.
    neighbourhood_function
        neighbourhood function for the SOM; only "gaussian" is supported.
    verbose
        whether to use verbose mode or not
    n_epochs
        number of training epochs, each of which updates the SOM once from all cells
    metric
        distance between cells and nodes
    device
        torch device to train on, e.g. "cpu", "cuda" or "mps"
    som_kwargs
        further keyword arguments for :class:`fastersom.SOM`, e.g. `sigma_final` or `deterministic`

    Returns
    -------
        trained SOM
    """
    from fastersom import SOM  # imports torch, so only when needed

    if learning_rate is not None:
        warnings.warn(
            "`learning_rate` has no effect with the batch SOM and will be removed in a future version.",
            FutureWarning,
            stacklevel=2,
        )
    if neighbourhood_function != "gaussian":
        raise ValueError('Unknown neighbourhood_function, only "gaussian" is supported')
    if weight_init not in _WEIGHT_INITS:
        raise ValueError('Unknown weight_init, must be one of "random" or "pca"')
    som = SOM(
        n_rows=som_dim[0],
        n_cols=som_dim[1],
        n_epochs=n_epochs,
        sigma_initial=sigma,
        init=_WEIGHT_INITS[weight_init],
        metric=metric,
        random_state=seed,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
        **(som_kwargs or {}),
    )
    logger.info("Training SOM")
    som.fit(x)
    return som


def kmeans_clustering(
    x: np.ndarray,
    n_clusters: int = 100,
    seed: int = 42,
    metric: Literal["euclidean", "cosine", "correlation"] = "euclidean",
    device: str = "cpu",
    batch_size: int | None = None,
    verbose: bool = False,
    kmeans_kwargs: dict | None = None,
) -> KMeans:
    """Cluster the given cytometry data into many small clusters with :class:`fastersom.KMeans`.

    Parameters
    ----------
    x
        cytometry data of shape `(n_samples, n_features)`
    n_clusters
        number of centroids
    seed
        random seed for reproducibility
    metric
        distance between cells and centroids
    device
        torch device to run on, e.g. "cpu", "cuda" or "mps"
    batch_size
        number of cells processed at once. Affects memory and speed only, not the result. If None, chosen by fastersom.
    verbose
        whether to use verbose mode or not
    kmeans_kwargs
        further keyword arguments for :class:`fastersom.KMeans`, e.g. `init` or `max_iter`

    Returns
    -------
        fitted k-means
    """
    from fastersom import KMeans  # imports torch, so only when needed

    kmeans = KMeans(
        n_clusters=n_clusters,
        metric=metric,
        random_state=seed,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
        **(kmeans_kwargs or {}),
    )
    logger.info("Fitting k-means")
    kmeans.fit(x)
    return kmeans


def _nodes(model: SOM | KMeans) -> tuple[np.ndarray, tuple[int, ...]]:
    """The node weights of a SOM or the centroids of k-means, and the shape of their meta-clustering."""
    if hasattr(model, "weights_"):
        return model.weights_, (model.n_rows, model.n_cols)
    return model.cluster_centers_, (model.n_clusters,)


def meta_clustering(
    som: SOM | KMeans,
    n_features: int,
    min_clusters: int = 2,
    max_clusters: int = 10,
    n_resamples: int = 100,
    resample_frac: float = 0.9,
    verbose: bool = False,
    agglomerative_clustering_kwargs: dict | None = None,
    rng: RNGLike | SeedLike | None = None,
) -> tuple[np.ndarray, ConsensusClustering]:
    """Meta-clustering of SOM nodes (or k-means centroids) using consensus clustering, as in FlowSOM (R).

    The nodes are resampled `n_resamples` times, each time a fraction `resample_frac` of them, and every resample is
    clustered hierarchically into k clusters. The consensus of two nodes is the fraction of the resamples containing
    both in which they fall into the same cluster. The meta-clusters are the average-linkage hierarchical clustering of
    one minus the consensus, cut into k clusters, as `metaClustering_consensus` of FlowSOM (R) with ConsensusClusterPlus
    does. The consensus matrices are computed with the `consensusclustering` package
    (https://github.com/burtonrj/consensusclustering), which also chooses k between `min_clusters` and `max_clusters`
    by the knee of the area under the cumulative distribution of the consensus.

    Parameters
    ----------
    som
        trained SOM or fitted k-means
    n_features
        number of features in the cytometry data
    min_clusters
        minimum number of clusters to consider
    max_clusters
        maximum number of clusters to consider; k is fixed if it equals `min_clusters`
    n_resamples
        number of resamples for consensus clustering
    resample_frac
        fraction of the nodes in each resample. Defaults to 0.9 as in FlowSOM (R); the default was 0.5 before.
    verbose
        whether to use verbose mode or not
    agglomerative_clustering_kwargs
        keyword arguments for the `sklearn.cluster.AgglomerativeClustering` of each resample. If None, defaults to
        {"metric": "euclidean", "linkage": "average"}, as in FlowSOM (R). The final clustering of the consensus always
        uses average linkage.
    rng
        random number generator or seed for reproducibility

    Returns
    -------
    meta-clustering of SOM nodes and the consensus clustering object
    """
    agglomerative_clustering_kwargs = (
        agglomerative_clustering_kwargs
        if agglomerative_clustering_kwargs is not None
        else {"metric": "euclidean", "linkage": "average"}
    )
    cc = ConsensusClustering(
        AgglomerativeClustering(**agglomerative_clustering_kwargs),
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        n_resamples=n_resamples,
        resample_frac=resample_frac,
        rng=rng,
    )
    weights, shape = _nodes(som)
    flatten_weights = weights.reshape(-1, n_features)
    cc.fit(flatten_weights, progress_bar=verbose)
    k = min_clusters if min_clusters == max_clusters else cc.best_k()
    if k is None:
        k = cc.best_k(method="change_in_auc")
        warnings.warn(
            f"No knee found in the area under the consensus CDF; choosing k={k} by its largest change instead.",
            UserWarning,
            stacklevel=2,
        )
    distance = 1 - cc.consensus_k(k)
    np.fill_diagonal(distance, 0)  # nodes never resampled have no consensus, not even with themselves
    final = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    meta_class = final.fit_predict(distance)
    return meta_class.reshape(shape), cc


def leiden_meta_clustering(
    som: SOM | KMeans,
    resolution: float = 1.0,
    n_neighbors: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Meta-clustering of SOM nodes (or k-means centroids) using Leiden clustering of their nearest-neighbour graph.

    Each node is connected to its `n_neighbors` nearest other nodes, under the metric the SOM was trained with, by an
    unweighted, undirected edge. The modularity of this graph is optimised with the Leiden algorithm of
    `graspologic-native` (https://github.com/graspologic-org/graspologic-native), in 10 iterations that each start
    from the previous partition.

    Parameters
    ----------
    som
        trained SOM or fitted k-means
    resolution
        resolution of the Leiden clustering; higher values give more clusters
    n_neighbors
        number of nearest neighbours of each node in the graph (at most the number of nodes minus one)
    seed
        random seed for reproducibility

    Returns
    -------
    meta-clustering of SOM nodes
    """
    weights, shape = _nodes(som)
    n_nodes = len(weights)
    n_neighbors = min(n_neighbors, n_nodes - 1)
    if n_neighbors < 1:
        return np.zeros(shape, dtype=np.int64)
    graph = kneighbors_graph(weights, n_neighbors, metric=som.metric)
    graph = graph.maximum(graph.T).tocsr()  # undirected
    _, partition = gn.leiden_csr(
        graph.indptr.astype(np.int64),
        graph.indices.astype(np.int32),
        graph.data.astype(np.float64),
        n_nodes,
        resolution=resolution,
        iterations=10,
        seed=seed,
    )
    communities = np.array([partition[node] for node in range(n_nodes)])
    return np.unique(communities, return_inverse=True)[1].reshape(shape)


class ClusteringObjects(TypedDict):
    som: SOM | KMeans
    meta_class: np.ndarray
    consensus_clustering: ConsensusClustering | None


def flowsom_clustering(
    adata: AnnData,
    key_added: str = "clusters",
    som_dim: tuple[int, int] = (10, 10),
    sigma: float | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    seed: int = 42,
    weight_init: Literal["random", "pca"] = "random",
    neighbourhood_function: Literal["gaussian"] = "gaussian",
    min_clusters: int = 2,
    max_clusters: int = 10,
    n_resamples: int = 100,
    resample_frac: float = 0.9,
    inplace: bool = True,
    verbose: bool = False,
    agglomerative_clustering_kwargs: dict | None = None,
    return_clustering_objs: bool = False,
    n_epochs: int = 10,
    metric: Literal["euclidean", "manhattan", "cosine", "correlation"] = "euclidean",
    device: str = "cpu",
    som_kwargs: dict | None = None,
    meta_clustering_method: Literal["consensus", "leiden"] = "consensus",
    resolution: float = 1.0,
    n_neighbors: int = 5,
    method: Literal["som", "kmeans"] = "som",
    markers: Sequence[str] | None = None,
) -> AnnData | tuple[AnnData, ClusteringObjects]:
    """Cluster cytometry data using FlowSOM.

    Based on the original FlowSOM R package by Van Gassen et al. (2015)
    (https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.22625.) and the python
    implementation found at https://github.com/Hatchin/FlowSOM. The SOM is a batch SOM trained with
    fastersom (https://github.com/dschaub95/fastersom). With `method="kmeans"`, the centroids of k-means take the
    place of the SOM's nodes.

    Parameters
    ----------
    adata
        annotated data matrix of shape `(n_samples, n_features)`
    key_added
        key under which to add the cluster labels
    som_dim
        dimensions of the SOM; for k-means, the number of centroids is their product
    sigma
        initial width of the gaussian neighbourhood in grid units, which decays to a tenth of it (at most one grid
        unit). If None, defaults to a quarter of the shorter side of the SOM (rounded up to half a unit).
    learning_rate
        deprecated, has no effect: the batch SOM moves each node to the neighbourhood-weighted mean of the cells.
    batch_size
        number of cells processed at once. Affects memory and speed only, not the result. If None, chosen by fastersom.
    seed
        random seed for reproducibility
    weight_init
        weight initialization method: "random" starts from randomly chosen cells, "pca" from a regular grid in the
        plane of the first two principal components of the data.
    neighbourhood_function
        neighbourhood function for the SOM; only "gaussian" is supported.
    min_clusters
        minimum number of clusters to consider (consensus meta-clustering only)
    max_clusters
        maximum number of clusters to consider (consensus meta-clustering only)
    n_resamples
        number of resamples for consensus clustering
    resample_frac
        fraction of the SOM nodes in each resample of consensus clustering
    inplace
        whether to copy the AnnData object or modify it
    verbose
        whether to use verbose mode or not
    agglomerative_clustering_kwargs
        keyword arguments for the `sklearn.cluster.AgglomerativeClustering` of each resample in consensus clustering.
        If None, defaults to {"metric": "euclidean", "linkage": "average"}, as in FlowSOM (R). The final clustering of
        the consensus always uses average linkage.
    return_clustering_objs
        whether to return the clustering objects in addition to the annotated data matrix
    n_epochs
        number of SOM training epochs, each of which updates the SOM once from all cells
    metric
        distance between cells and SOM nodes, also used for the Leiden graph of the nodes; consensus
        meta-clustering takes its metric from `agglomerative_clustering_kwargs`
    device
        torch device to train the SOM on, e.g. "cpu", "cuda" or "mps"
    som_kwargs
        further keyword arguments for :class:`fastersom.SOM`, e.g. `sigma_final` or `deterministic`, or for
        :class:`fastersom.KMeans` with `method="kmeans"`
    meta_clustering_method
        how to cluster the SOM nodes: "consensus" for consensus hierarchical clustering as in FlowSOM (R) (see
        :func:`pytometry.tl.meta_clustering`), which chooses the number of clusters between `min_clusters` and
        `max_clusters`, or "leiden" for Leiden clustering of their nearest-neighbour graph
    resolution
        resolution of the Leiden clustering (Leiden meta-clustering only)
    n_neighbors
        number of nearest neighbours of each node in the Leiden graph (Leiden meta-clustering only)
    method
        "som" for the SOM of FlowSOM, or "kmeans" for k-means centroids instead of its nodes. k-means has no
        `sigma`, `n_epochs`, `weight_init` or manhattan `metric`.
    markers
        markers (`var_names`) to cluster on, e.g. lineage markers without functional ones. All markers if None.

    Returns
    -------
    annotated data matrix with cluster labels added under `key_added`, the SOM node of every cell under
    `f"{key_added}_som_node"` (-1 for cells without one) and the SOM's weights and meta-clusters under
    `adata.uns[key_added]`, as used by :func:`pytometry.pl.flowsom_stars` and the other FlowSOM plots.
    If `return_clustering_objs` is True, a tuple containing the annotated data matrix
    and the clustering objects is returned.

    """
    if meta_clustering_method not in ("consensus", "leiden"):
        raise ValueError('Unknown meta_clustering_method, must be one of "consensus" or "leiden"')
    if method not in ("som", "kmeans"):
        raise ValueError('Unknown method, must be one of "som" or "kmeans"')
    adata = adata if inplace else adata.copy()
    var_names = adata.var_names if markers is None else pd.Index(markers)
    x = adata.X if markers is None else adata[:, var_names].X
    logger.info("Running FlowSOM clustering")
    if method == "som":
        som = som_clustering(
            x,
            som_dim=som_dim,
            sigma=sigma,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
            weight_init=weight_init,
            neighbourhood_function=neighbourhood_function,
            verbose=verbose,
            n_epochs=n_epochs,
            metric=metric,
            device=device,
            som_kwargs=som_kwargs,
        )
    else:
        som = kmeans_clustering(
            x,
            n_clusters=som_dim[0] * som_dim[1],
            seed=seed,
            metric=metric,
            device=device,
            batch_size=batch_size,
            verbose=verbose,
            kmeans_kwargs=som_kwargs,
        )
    logger.info("Meta-clustering of SOM nodes")
    cc = None
    if meta_clustering_method == "consensus":
        meta_class, cc = meta_clustering(
            som,
            x.shape[1],
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            n_resamples=n_resamples,
            resample_frac=resample_frac,
            verbose=verbose,
            agglomerative_clustering_kwargs=agglomerative_clustering_kwargs,
            rng=seed,
        )
    else:
        meta_class = leiden_meta_clustering(som, resolution=resolution, n_neighbors=n_neighbors, seed=seed)
    logger.info("Assigning cluster labels to cells")
    # som.labels_ holds the node (or centroid) of every cell, -1 for cells without one (all-zero cells under cosine, constant
    # cells under correlation), which get no cluster
    node_labels = np.array([f"cluster_{c}" for c in meta_class.ravel()], dtype=object)
    labels = node_labels[som.labels_]
    labels[som.labels_ == -1] = None
    adata.obs[key_added] = pd.Categorical(labels, categories=natsorted(set(node_labels)))
    adata.obs[f"{key_added}_som_node"] = som.labels_
    adata.uns[key_added] = {
        "params": {
            "method": method,
            "som_dim": list(som_dim),
            "metric": metric,
            "meta_clustering_method": meta_clustering_method,
        },
        "weights": _nodes(som)[0],
        "meta_class": meta_class,
        "var_names": np.asarray(var_names),
    }
    if return_clustering_objs:
        return adata, ClusteringObjects(som=som, meta_class=meta_class, consensus_clustering=cc)
    return adata
