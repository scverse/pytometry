import pytest
from anndata import AnnData
from sklearn.datasets import make_classification
from sklearn.metrics import fowlkes_mallows_score
from sklearn.preprocessing import MinMaxScaler

from pytometry.tools.clustering import flowsom_clustering


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


def test_flowsom_clustering(example_data):
    example_data = flowsom_clustering(
        example_data,
        min_clusters=2,
        max_clusters=10,
        verbose=True,
        key_added="flowsom_labels",
        seed=42,
        som_dim=(10, 10),
        sigma=1.0,
    )
    assert example_data.obs["flowsom_labels"].nunique() == 5
    score = fowlkes_mallows_score(example_data.obs["ground_truth"], example_data.obs["flowsom_labels"])
    assert score > 0.9
