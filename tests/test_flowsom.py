import numpy as np
import pytest
from anndata import AnnData
from sklearn.datasets import make_classification
from sklearn.metrics import fowlkes_mallows_score
from sklearn.preprocessing import MinMaxScaler

from pytometry.tl.clustering import flowsom_clustering
from pytometry.tl.clustering._flowsom import som_clustering


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
    adata.obs["ground_truth"] = y
    return adata


@pytest.mark.parametrize("meta_clustering_method", ["consensus", "leiden"])
@pytest.mark.parametrize("weight_init", ["random", "pca"])
def test_flowsom_clustering(example_data, meta_clustering_method, weight_init):
    example_data = flowsom_clustering(
        example_data,
        min_clusters=2,
        max_clusters=10,
        verbose=True,
        key_added="flowsom_labels",
        seed=0,
        som_dim=(10, 10),
        weight_init=weight_init,
        meta_clustering_method=meta_clustering_method,
    )
    assert example_data.obs["flowsom_labels"].nunique() == 5
    score = fowlkes_mallows_score(example_data.obs["ground_truth"], example_data.obs["flowsom_labels"])
    assert score > 0.9


def test_flowsom_clustering_objects(example_data):
    adata, objs = flowsom_clustering(example_data, inplace=False, return_clustering_objs=True)
    assert "clusters" not in example_data.obs
    som = objs["som"]
    assert objs["meta_class"].shape == (10, 10)
    assert som.weights_.shape == (100, example_data.n_vars)
    labels = [f"cluster_{c}" for c in objs["meta_class"].ravel()[som.predict(example_data.X)]]
    assert list(adata.obs["clusters"]) == labels
    assert objs["consensus_clustering"] is not None
    _, objs = flowsom_clustering(example_data, meta_clustering_method="leiden", return_clustering_objs=True)
    assert objs["consensus_clustering"] is None


def test_som_uses_all_cells(example_data):
    # cells sorted by class, so that the first cells all come from one class: the SOM must fit every class equally well
    classes = example_data.obs["ground_truth"].to_numpy()
    order = np.argsort(classes, kind="stable")
    x, classes = example_data.X[order], classes[order]
    som = som_clustering(x, seed=0)
    error = np.linalg.norm(x - som.weights_[som.labels_], axis=1)
    error_per_class = [error[classes == c].mean() for c in np.unique(classes)]
    assert max(error_per_class) < 2 * min(error_per_class)


def test_som_clustering_arguments(example_data):
    with pytest.warns(FutureWarning, match="learning_rate"):
        som_clustering(example_data.X, learning_rate=0.5, n_epochs=1)
    with pytest.raises(ValueError, match="neighbourhood_function"):
        som_clustering(example_data.X, neighbourhood_function="bubble")
    with pytest.raises(ValueError, match="weight_init"):
        som_clustering(example_data.X, weight_init="sample")
    som = som_clustering(example_data.X, som_dim=(4, 6), metric="cosine", som_kwargs={"sigma_final": 0.5})
    assert (som.n_rows, som.n_cols, som.metric, som.sigma_final) == (4, 6, "cosine", 0.5)
    assert som.device == "cpu"


@pytest.mark.parametrize("layout", ["mst", "grid"])
def test_flowsom_plots(example_data, layout):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pytometry.pl import flowsom_markers, flowsom_pies, flowsom_stars

    example_data.var_names = [f"marker_{i}" for i in range(example_data.n_vars)]
    example_data.obs["ground_truth"] = example_data.obs["ground_truth"].astype(str).astype("category")
    flowsom_clustering(example_data, som_dim=(6, 8), seed=0)
    assert example_data.obs["clusters_som_node"].between(0, 47).all()
    assert example_data.uns["clusters"]["weights"].shape == (48, example_data.n_vars)
    ax = flowsom_stars(example_data, layout=layout)
    assert len(ax.collections) >= 3
    flowsom_stars(example_data, markers=["marker_0", "marker_2"], layout=layout, node_size=None)
    fig = flowsom_markers(example_data, ["marker_0", "marker_1", "marker_2"], layout=layout, n_cols=2)
    assert len(fig.axes) == 4 + 3  # 2 x 2 panels and a colorbar per marker
    flowsom_pies(example_data, "ground_truth", layout=layout)
    with pytest.raises(KeyError, match="marker_9"):
        flowsom_stars(example_data, markers=["marker_9"], layout=layout)
    with pytest.raises(KeyError, match="No FlowSOM result"):
        flowsom_stars(example_data, key="other")
    plt.close("all")


def test_flowsom_plots_edge_cases(example_data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pytometry.pl import flowsom_stars

    flowsom_clustering(example_data, seed=0)
    assert "mst_layout" not in example_data.uns["clusters"]
    flowsom_stars(example_data)
    assert "mst_layout" in example_data.uns["clusters"]
    # a subset without the cells of cluster_0, which anndata then drops from the categories
    subset = example_data[example_data.obs["clusters"] != "cluster_0"]
    flowsom_stars(subset)
    single = flowsom_clustering(example_data.copy(), som_dim=(1, 1), meta_clustering_method="leiden")
    flowsom_stars(single)
    plt.close("all")


def test_meta_clustering_single_k(example_data):
    adata = flowsom_clustering(example_data, min_clusters=3, max_clusters=3, seed=0)
    assert adata.obs["clusters"].nunique() == 3


@pytest.mark.parametrize("meta_clustering_method", ["consensus", "leiden"])
def test_flowsom_clustering_kmeans(example_data, meta_clustering_method):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pytometry.pl import flowsom_pies, flowsom_stars

    # the consensus knee picks 4 clusters for these centroids, so k is fixed to test the rest of the pipeline
    adata, objs = flowsom_clustering(
        example_data,
        method="kmeans",
        meta_clustering_method=meta_clustering_method,
        min_clusters=5,
        max_clusters=5,
        seed=0,
        return_clustering_objs=True,
    )
    assert objs["som"].cluster_centers_.shape == (100, example_data.n_vars)
    assert objs["meta_class"].shape == (100,)
    assert fowlkes_mallows_score(adata.obs["ground_truth"], adata.obs["clusters"]) > 0.9
    flowsom_stars(adata)
    adata.obs["ground_truth"] = adata.obs["ground_truth"].astype(str).astype("category")
    flowsom_pies(adata, "ground_truth")
    with pytest.raises(ValueError, match="no grid"):
        flowsom_stars(adata, layout="grid")
    plt.close("all")
