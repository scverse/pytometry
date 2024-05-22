import anndata
import numpy
import pandas
import readfcs

from pytometry.preprocessing import _dummy_spillover, compensate, create_comp_mat
from pytometry.read_write import read_fcs
from pytometry.tools import normalize_arcsinh, normalize_biExp, normalize_logicle


# test read function
def test_read_fcs():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data, reindex=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


# test reindex false
def test_read_fcs2():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data, reindex=False)
    assert isinstance(adata, anndata._core.anndata.AnnData)


# test compensate
def test_compensate():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    compensate(adata)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_compensate_inplace():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    adata2 = compensate(adata, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_spill():
    n_rows = 12
    spillmat = _dummy_spillover(n_rows=n_rows)
    assert isinstance(spillmat, pandas.DataFrame)
    assert spillmat.shape == (n_rows, n_rows)


def test_create_comp_mat():
    n_rows = 5
    spillmat = _dummy_spillover(n_rows=n_rows)
    comp_mat = create_comp_mat(spillmat)
    assert numpy.sum(comp_mat.values) == n_rows * 0.5


# test custom dummy compensation
def test_compensate2():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    # use extra compensation matrix
    spillmat = _dummy_spillover(
        n_rows=adata.uns["meta"]["spill"].shape[0],
        row_names=adata.uns["meta"]["spill"].index,
    )
    adata2 = compensate(adata, comp_matrix=spillmat, matrix_type="spillover", inplace=False)
    assert adata2.X.sum() != adata.X.sum()


# test return types for normalization
def test_normalize_arcsinh():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    normalize_arcsinh(adata, cofactor=1, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_arcsinh2():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    adata2 = normalize_arcsinh(adata, cofactor=1, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_normalize_arcsinh3():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    cofactor = pandas.Series(numpy.repeat(1, adata.n_vars), index=adata.var_names)
    adata2 = normalize_arcsinh(adata, cofactor=cofactor, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_normalize_biexp():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    normalize_biExp(adata, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_biexp2():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    adata2 = normalize_biExp(adata, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_normalize_logicle():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    normalize_logicle(adata, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_logicle2():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    adata2 = normalize_logicle(adata, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)
