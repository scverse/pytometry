import anndata
import numpy
import pandas
from flowutils import transforms

from pytometry.io import read_fcs
from pytometry.pp import _dummy_spillover, compensate, create_comp_mat
from pytometry.tl import normalize_arcsinh, normalize_autologicle, normalize_biexp, normalize_logicle


# test read function
def test_read_fcs(oetjen18_t1):
    adata = read_fcs(oetjen18_t1, reindex=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


# test reindex false
def test_read_fcs2(oetjen18_t1):
    adata = read_fcs(oetjen18_t1, reindex=False)
    assert isinstance(adata, anndata._core.anndata.AnnData)


# test read function with ignore_offset_error=True
# TODO: find available FCS file with offset error
def test_read_fcs_with_offset_error(oetjen18_t1):
    adata = read_fcs(oetjen18_t1, reindex=True, ignore_offset_error=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


# test compensate
def test_compensate(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    compensate(adata)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_compensate_inplace(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
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
def test_compensate2(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    # use extra compensation matrix
    spillmat = _dummy_spillover(
        n_rows=adata.uns["meta"]["spill"].shape[0],
        row_names=adata.uns["meta"]["spill"].index,
    )
    adata2 = compensate(adata, comp_matrix=spillmat, matrix_type="spillover", inplace=False)
    assert adata2.X.sum() != adata.X.sum()


# test return types for normalization
def test_normalize_arcsinh(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    normalize_arcsinh(adata, cofactor=1, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_arcsinh2(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    adata2 = normalize_arcsinh(adata, cofactor=1, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_normalize_arcsinh3(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    cofactor = pandas.Series(numpy.repeat(1, adata.n_vars), index=adata.var_names)
    adata2 = normalize_arcsinh(adata, cofactor=cofactor, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_normalize_biexp(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    normalize_biexp(adata, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_biexp2(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    adata2 = normalize_biexp(adata, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_normalize_biexp3(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    params = {
        name: pandas.Series(numpy.repeat(value, adata.n_vars), index=adata.var_names)
        for name, value in {"negative": 0.0, "width": -10.0, "positive": 4.41854, "max_value": 262144.0}.items()
    }
    adata2 = normalize_biexp(adata, inplace=False, **params)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_normalize_logicle(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    normalize_logicle(adata, inplace=True)
    assert isinstance(adata, anndata._core.anndata.AnnData)


def test_normalize_logicle2(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
    adata2 = normalize_logicle(adata, inplace=False)
    assert isinstance(adata2, anndata._core.anndata.AnnData)


def test_autologicle_param_override(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
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


def test_return_params(oetjen18_t1):
    adata = read_fcs(oetjen18_t1)
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
    adata = read_fcs(oetjen18_t1)
    # case 4: mutative, don't return params
    result = normalize_autologicle(adata, inplace=True, return_params=False)
    assert (adata.X == adata2.X).all()  # check if inplace=True works
    assert result is None
