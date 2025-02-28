import anndata
import numpy
import pandas
import readfcs
from flowutils import transforms

from pytometry.io import read_fcs
from pytometry.pp import _dummy_spillover, compensate, create_comp_mat
from pytometry.tl import normalize_arcsinh, normalize_autologicle, normalize_biexp, normalize_logicle


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
    normalize_biexp(adata, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_biexp2():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    adata2 = normalize_biexp(adata, inplace=False)
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


def test_autologicle_param_override():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    channel_indices = list(range(adata.n_vars))
    params = {
        "w": 0.1,
        "t": 4000,
        "m": 4.5,
        "a": 0,
    }
    params_list = [params for _ in range(adata.n_vars)]  # list of identical params
    result1 = transforms.logicle(adata.X, channel_indices, **params)
    result2 = normalize_autologicle(adata, inplace=False, params_override=params_list).X
    assert (result1 == result2).all()


def test_return_params():
    path_data = readfcs.datasets.Oetjen18_t1()
    adata = read_fcs(path_data)
    # case 1: non-mutative, return params
    adata2, params_list = normalize_autologicle(adata, inplace=False, return_params=True)
    assert isinstance(adata2, anndata._core.anndata.AnnData)
    assert isinstance(params_list, list)
    assert isinstance(params_list[0], dict)
    # case 2: non-mutative, don't return params
    adata2 = normalize_autologicle(adata, inplace=False, return_params=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)
    # case 3: mutative, return params
    params_list = normalize_autologicle(adata, inplace=True, return_params=True)
    assert isinstance(params_list, list)
    assert isinstance(params_list[0], dict)
    assert (adata.X == adata2.X).all()  # check if inplace=True works
    adata = read_fcs(path_data)
    # case 4: mutative, don't return params
    result = normalize_autologicle(adata, inplace=True, return_params=False)
    assert (adata.X == adata2.X).all() # check if inplace=True works
    assert result is None