import re
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

# import getpass
# import os.path


def create_comp_mat(spillmat: pd.DataFrame, relevant_data: str = "") -> pd.DataFrame:
    """Creates a compensation matrix from a spillover matrix.

    Parameters
    ----------
    spillmat
        Spillover matrix as pandas dataframe.
    relevant_data:
        A list of channels for customized selection.

    Returns
    -------
    DataFrame of the compensation matrix.
    """
    comp_mat = np.linalg.inv(spillmat)

    if relevant_data == "":
        compens = pd.DataFrame(comp_mat, columns=list(spillmat.columns))
    else:
        compens = pd.DataFrame(comp_mat, columns=relevant_data)

    return compens


def find_indexes(
    adata: AnnData,
    var_key: str = None,
    key_added: str = "signal_type",
    data_type: Literal["facs", "cytof"] = "facs",
    inplace: bool = True,
) -> AnnData | None:
    """Find channels of interest for computing compensation.

    Parameters
    ----------
    adata
        AnnData object.
    var_key
        Key where to check if a feature is an area, height etc. type of value. Use `var_names` if None.
    key_added
        Key where result vector is added to the adata.var.
    data_type
        Type of cytometry data
    inplace

    Returns
    -------
    Depending on `inplace`, returns or updates `adata` with the following updated field `adata.var[f'{key_added}']`
    """
    adata = adata if inplace else adata.copy()

    if var_key is None:
        index = adata.var.index
    elif var_key in adata.var_keys():
        index = adata.var[var_key]
    else:
        raise KeyError(f"Did not find '{var_key}' in `.var_keys()`.")

    index_array = []

    if data_type.lower() == "facs":
        for item in index:
            item = item.upper()
            # find FSC and SSC channels first
            if re.match("(F|S)SC", item) is not None:
                index_array.append("other")
            elif item.endswith("-A"):
                index_array.append("area")
            elif item.endswith("-H"):
                index_array.append("height")
            else:
                index_array.append("other")

    elif data_type.lower() == "cytof":
        for item in index:
            if item.endswith("Di") or item.endswith("Dd"):
                index_array.append("element")
            else:
                index_array.append("other")
    else:
        print(f"{data_type} not recognized. Must be either 'facs' or                'cytof'")
    adata.var[key_added] = pd.Categorical(index_array)
    return None if inplace else adata


# rename compute bleedthr to compensate
def compensate(
    adata: AnnData,
    comp_matrix: pd.DataFrame = None,
    matrix_type: str = "spillover",
    inplace: bool = True,
) -> AnnData | None:
    """Computes compensation for data channels.

    Parameters
    ----------
    adata
        AnnData object.
    key
        Key where result vector is added to the adata.var.
    comp_matrix
        A custom compensation matrix. Please note that by default we use the spillover matrix directly for numeric stability.
    matrix_type
        Whether to use a spillover matrix (default) or a compensation matrix. Only considered for custom compensation
        matrices. Usually, custom compensation matrices are the inverse of the spillover matrix. If you want to use
        a compensation matrix, not the spillover matrix, set `matrix_type` to `compensation`.
    inplace
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `inplace`, returns or updates `adata`
    """
    adata = adata if inplace else adata.copy()

    # locate compensation matrix
    if comp_matrix is not None:
        if matrix_type == "spillover":
            compens = comp_matrix
        elif matrix_type == "compensation":
            compens = create_comp_mat(comp_matrix)
        else:
            raise KeyError(f"Expected 'spillover' or 'compensation' as `matrix_type`, but got '{matrix_type}'.")
        # To Do: add checks that this input is correct
    elif adata.uns["meta"]["spill"] is not None:
        compens = adata.uns["meta"]["spill"]
    else:
        raise KeyError(f"Did not find .uns['meta']['spill'] nor '{comp_matrix}'.")

    # save original data as layer
    if "original" not in adata.layers:
        adata.layers["original"] = adata.X.copy()

    # Ignore channels 'FSC-H', 'FSC-A', 'SSC-H', 'SSC-A',
    # 'FSC-Width', 'Time'
    # and compensate only the values indicated in the compensation matrix
    # Note:
    # the compensation matrix may have different index names than the adata.X matrix
    ref_col = adata.var.index
    idx_in = np.intersect1d(compens.columns, ref_col)
    if not idx_in.any():
        # try the adata.var['channel'] as reference
        ref_col = adata.var["channel"]
        idx_in = np.intersect1d(compens.columns, ref_col)
        if not idx_in.any():
            raise ValueError(
                "Could not match the column names of the compensation matrix"
                'with neither `adata.var.index` nor `adata.var["channel"].'
            )
    # match columns of spill mat such that they exactly correspond to adata.var.index
    ref_names = ref_col[np.in1d(ref_col, idx_in)]
    query_names = compens.columns[np.in1d(compens.columns, idx_in)]
    idx_sort = [np.where(query_names == x)[0][0] for x in ref_names]
    query_idx = np.in1d(compens.columns, query_names)
    ref_idx = np.in1d(ref_col, ref_names)

    # subset compensation matrix to the columns to run the compensation on
    compens = compens.iloc[query_idx, query_idx]
    # sort compensation matrix by adata.var_names
    compens = compens.iloc[idx_sort, idx_sort]
    X_comp = np.linalg.solve(compens.T, adata.X[:, ref_idx].T).T

    X = adata.X.copy()
    adata.X[:, ref_idx] = X_comp

    if np.array_equal(X, adata.X):
        print("Compensation failed - matrices before and after are equivalent. Please check your compensation matrix.")
    del X

    # check for nan values
    nan_val = np.isnan(adata.X[:, ref_idx]).sum()
    if nan_val > 0:
        print(f"{nan_val} NaN values found after compensation. Please adjust compensation matrix.")

    return None if inplace else adata


def split_signal(
    adata: AnnData,
    var_key: str | None = None,
    key: str = "signal_type",
    option: str = "area",
    data_type: Literal["facs", "cytof"] = "facs",
    inplace: bool = True,
) -> AnnData | None:
    """Method to filter out height or area data.

    Parameters
    ----------
    adata
        AnnData object containing data.
    var_key
        Key where to check if a feature is an area, height etc. type of value. Use `var_names` if None.
    key
        Key for adata.var where the variable type is stored.
    option
        For choosing 'area' or 'height' in case of FACS data and 'element' for cyTOF data.
    data_type
        Type of cytometry data
    inplace
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `inplace`, returns or updates `adata` with area or height data in `.var`
    """
    adata = adata if inplace else adata.copy()

    option_key = option
    key_in = key

    possible_options = ["area", "height", "other", "element"]

    if option_key not in possible_options:
        print(f"'{option_key}' is not a valid category. Return all.")
        return None if inplace else adata
    # Check if indices for area and height have been computed
    if key_in not in adata.var_keys():
        find_indexes(adata, var_key=var_key, data_type=data_type)

    indx = adata.var[key_in] == option_key
    # if none of the indices is true, abort
    if sum(indx) < 1:
        print(f"'{option_key}' is not in adata.var['{key_in}']. Return all.")
        return None if inplace else adata

    non_idx = np.flatnonzero(np.invert(indx))

    # merge non-idx entries in data matrix with obs
    non_cols = adata.var_names[non_idx].values
    for idx, colname in enumerate(non_cols):
        if colname == "":
            colname = adata.var["channel"][non_idx[idx]]
        adata.obs[colname] = adata.X[:, non_idx[idx]].copy()

    # subset the anndata object
    adata._inplace_subset_var(indx)

    return None if inplace else adata


# create test compensation matrix
def _dummy_spillover(n_rows: int = 10, row_names: pd.Index | list[str] | None = None) -> pd.DataFrame:
    """Create dummy spillover matrix for testing.

    Parameters
    ----------
    n_rows
        number of rows and columns.
    row_names
        Index to use for the resulting dataframe. Also used as column names.

    Returns
    -------
    A dummy spillover matrix with 2's on the diagonal
    """
    tmp_mat = np.diag(np.ones(n_rows) * 2)
    dummy_spill = pd.DataFrame(data=tmp_mat, index=row_names, columns=row_names)
    return dummy_spill
