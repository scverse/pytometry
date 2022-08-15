import anndata
import readfcs

from pytometry.read_write import read_fcs
from pytometry.tools import normalize_arcsinh, normalize_biExp, normalize_logicle


def test_read_fcs():
    path_data = readfcs.datasets.example()
    adata = read_fcs(path_data)
    assert isinstance(adata, anndata._core.anndata.AnnData)


# test return types
def test_normalize_arcsinh():
    path_data = readfcs.datasets.example()
    adata = read_fcs(path_data)
    normalize_arcsinh(adata, cofactor=1, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_arcsinh2():
    path_data = readfcs.datasets.example()
    adata = read_fcs(path_data)
    adata2 = normalize_arcsinh(adata, cofactor=1, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_normalize_biexp():
    path_data = readfcs.datasets.example()
    adata = read_fcs(path_data)
    normalize_biExp(adata, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_biexp2():
    path_data = readfcs.datasets.example()
    adata = read_fcs(path_data)
    adata2 = normalize_biExp(adata, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_normalize_logicle():
    path_data = readfcs.datasets.example()
    adata = read_fcs(path_data)
    normalize_logicle(adata, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_logicle2():
    path_data = readfcs.datasets.example()
    adata = read_fcs(path_data)
    adata2 = normalize_logicle(adata, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)
