from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from sklearn.preprocessing import MinMaxScaler

from pytometry.pl import flowsom_markers, flowsom_pies, flowsom_stars
from pytometry.tl.clustering import flowsom_clustering
from pytometry.tl.clustering._flowsom import meta_clustering, som_clustering


@pytest.fixture
def example_data() -> AnnData:
    x, y = make_classification(
        n_samples=10000,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=5,
        n_clusters_per_class=1,
        weights=None,
        random_state=42,
        class_sep=10.0,
    )
    adata = AnnData(MinMaxScaler().fit_transform(x))
    adata.var_names = [f"marker_{i}" for i in range(adata.n_vars)]
    adata.obs["ground_truth"] = y.astype(str)
    return adata


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_flowsom_clustering(example_data):
    example_data = flowsom_clustering(
        example_data,
        min_clusters=2,
        max_clusters=10,
        verbose=True,
        key_added="flowsom_labels",
        seed=0,
        som_dim=(10, 10),
    )
    assert example_data.obs["flowsom_labels"].nunique() == 5
    score = fowlkes_mallows_score(example_data.obs["ground_truth"], example_data.obs["flowsom_labels"])
    assert score > 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_init": "pca"},
        {"meta_clustering_method": "leiden"},
        {"method": "kmeans", "min_clusters": 5, "max_clusters": 5},
        {"method": "kmeans", "meta_clustering_method": "leiden"},
    ],
    ids=["pca", "leiden", "kmeans", "kmeans-leiden"],
)
def test_flowsom_clustering_options(example_data, kwargs):
    adata = flowsom_clustering(example_data, seed=0, **kwargs)
    assert fowlkes_mallows_score(adata.obs["ground_truth"], adata.obs["clusters"]) > 0.9


def test_flowsom_clustering_outputs(example_data):
    adata, objs = flowsom_clustering(example_data, inplace=False, return_clustering_objs=True)
    assert "clusters" not in example_data.obs
    nodes = adata.obs["clusters_som_node"].to_numpy()
    assert (nodes == objs["som"].predict(example_data.X)).all()
    assert list(adata.obs["clusters"]) == [f"cluster_{c}" for c in objs["meta_class"].ravel()[nodes]]
    assert adata.uns["clusters"]["weights"].shape == (100, example_data.n_vars)


def test_som_uses_all_cells(example_data):
    # cells sorted by class, so that the first cells all come from one class: the SOM must fit every class equally well
    classes = example_data.obs["ground_truth"].to_numpy()
    order = np.argsort(classes, kind="stable")
    x, classes = example_data.X[order], classes[order]
    som = som_clustering(x, seed=0)
    error = np.linalg.norm(x - som.weights_[som.labels_], axis=1)
    error_per_class = [error[classes == c].mean() for c in np.unique(classes)]
    assert max(error_per_class) < 2 * min(error_per_class)


def test_learning_rate_deprecated(example_data):
    with pytest.warns(FutureWarning, match="learning_rate"):
        som_clustering(example_data.X, learning_rate=0.5, n_epochs=1)


def _centroids(codes: np.ndarray) -> SimpleNamespace:
    """Stand-in for fitted k-means with the given centroids, which is all that meta-clustering reads."""
    return SimpleNamespace(cluster_centers_=codes, n_clusters=len(codes))


def test_meta_clustering_uses_consensus():
    codes = np.random.default_rng(19).normal(size=(30, 2))
    meta, cc = meta_clustering(_centroids(codes), 2, min_clusters=4, max_clusters=4, rng=0)
    distance = 1 - cc.consensus_k(4)
    np.fill_diagonal(distance, 0)
    consensus = AgglomerativeClustering(n_clusters=4, metric="precomputed", linkage="average").fit_predict(distance)
    assert adjusted_rand_score(meta, consensus) == 1
    assert (meta == meta_clustering(_centroids(codes), 2, min_clusters=4, max_clusters=4, rng=0)[0]).all()


@pytest.mark.parametrize(("min_clusters", "max_clusters"), [(4, 4), (2, 8)], ids=["fixed-k", "chosen-k"])
def test_meta_clustering_separated_groups(min_clusters, max_clusters):
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(4), 10)
    codes = 10 * rng.normal(size=(4, 3))[groups] + rng.normal(scale=0.3, size=(40, 3))
    meta, _ = meta_clustering(_centroids(codes), 3, min_clusters=min_clusters, max_clusters=max_clusters, rng=0)
    assert adjusted_rand_score(meta, groups) == 1


def test_flowsom_clustering_markers(example_data):
    noise = np.random.default_rng(0).random((example_data.n_obs, 3))
    adata = AnnData(np.hstack([example_data.X, noise]), obs=example_data.obs)
    adata.var_names = [*example_data.var_names, "noise_0", "noise_1", "noise_2"]
    flowsom_clustering(adata, markers=list(example_data.var_names), seed=0)
    assert list(adata.uns["clusters"]["var_names"]) == list(example_data.var_names)
    assert fowlkes_mallows_score(adata.obs["ground_truth"], adata.obs["clusters"]) > 0.9


@pytest.mark.parametrize("layout", ["mst", "grid"])
def test_flowsom_plots(example_data, layout):
    flowsom_clustering(example_data, seed=0)
    flowsom_stars(example_data, layout=layout)
    assert flowsom_markers(example_data, ["marker_0", "marker_1", "marker_2"], layout=layout, n_cols=2).shape == (2, 2)
    flowsom_pies(example_data, "ground_truth", layout=layout)
    # a subset without cluster_0, which anndata drops from the categories
    flowsom_stars(example_data[example_data.obs["clusters"] != "cluster_0"], layout=layout)
    with pytest.raises(KeyError, match="No FlowSOM result"):
        flowsom_stars(example_data, key="other")
