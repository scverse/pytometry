import math
import re

import numpy as np
import pandas as pd
import seaborn as sb

# import FlowCytometryTools as fct
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib import rcParams

from ..tools import normalize_arcsinh

# import getpass
# import os.path


def create_comp_mat(spillmat, relevant_data=""):
    """Creates a compensation matrix from a spillover matrix.

    Args:
        spillmat (pd.DataFrame): Spillover matrix as pandas dataframe.
        relevant_data (str, optional):A list of channels for customized selection.
            Defaults to ''.

    Returns:
        pd.DataFrame: Compensation matrix as pandas dataframe.
    """
    if relevant_data == "":
        comp_mat = np.linalg.inv(spillmat)
        compens = pd.DataFrame(comp_mat, columns=list(spillmat.columns))
    else:
        comp_mat = np.linalg.inv(spillmat)
        compens = pd.DataFrame(comp_mat, columns=relevant_data)

    return compens


def find_indexes(
    adata: AnnData,
    var_key=None,
    key_added="signal_type",
    data_type="facs",
    copy: bool = False,
):
    """Find channels of interest for computing compensation.

    Args:
        adata (AnnData): anndata object
        var_key (str, optional): key where to check if a feature is an area,
             height etc. type of value. Use `var_names` if None.
        key_added (str, optional): key where result vector is added to the adata.var.
            Defaults to 'signal_type'.
        data_type (str, optional): either 'facs' or 'cytof'.
            Defaults to 'facs'.
        copy (bool, optional): Return a copy instead of writing to adata.
            Defaults to False.

    Returns:
        Depending on `copy`, returns or updates `adata` with the following updated field
            adata.var[f'{key_added}']
    """
    adata = adata.copy() if copy else adata

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
        print(
            f"{data_type} not recognized. Must be either 'facs' or               "
            " 'cytof'"
        )
    adata.var["signal_type"] = pd.Categorical(index_array)
    return adata if copy else None


# rename compute bleedthr to compensate
def compensate(
    adata: AnnData,
    var_key=None,
    key="signal_type",
    comp_matrix=None,
    copy: bool = False,
):
    """Computes compensation for data channels.

    Args:
        adata (AnnData): AnnData object
        var_key (str, optional): key where to check if a feature is an area,
             height etc. type of value. Use `var_names` if None.
        key (str, optional): key where result vector is added
            to the adata.var. Defaults to 'signal_type'.
        comp_matrix (None, optional): a custom compensation matrix.
        copy (bool, optional): Return a copy instead of writing to adata.
            Defaults to False.

    Returns:
        Depending on `copy`, returns or updates `adata`
    """
    adata = adata.copy() if copy else adata

    key_in = key

    # locate compensation matrix
    if comp_matrix is not None:
        compens = comp_matrix
        # To Do: add checks that this input is correct
    if adata.uns["meta"]["spill"] is not None:
        compens = create_comp_mat(adata.uns["meta"]["spill"])
    else:
        raise KeyError(f"Did not find .uns['meta']['spill'] nor '{comp_matrix}'.")

    # save original data as layer
    if "original" not in adata.layers:
        adata.layers["original"] = adata.X

    # Ignore channels 'FSC-H', 'FSC-A', 'SSC-H', 'SSC-A',
    # 'FSC-Width', 'Time'
    if key_in not in adata.var_keys():
        adata = find_indexes(adata, var_key=var_key, data_type="facs")
    # select non other indices
    indexes = np.invert(adata.var[key_in] == "other")

    # To Do:
    # the compensation matrix may have different index names than the adata.X matrix
    # add a check and match for the compensation
    X_comp = np.dot(adata.X[:, indexes], compens)
    adata.X[:, indexes] = X_comp
    return adata if copy else None


def split_signal(
    adata: AnnData,
    var_key=None,
    key="signal_type",
    option="area",
    data_type="facs",
    copy: bool = False,
):
    """Method to filter out height or area data.

    Args:
        adata (AnnData): AnnData object containing data.
        var_key (str, optional): key where to check if a feature is an area,
             height etc. type of value. Use `var_names` if None.
        key (str, optional): key for adata.var where the variable type is stored.
            Defaults to 'signal_type'.
        option (str, optional):  for choosing 'area' or 'height' in case of FACS data
            and 'element' for cyTOF data. Defaults to 'area'.
        data_type (str, optional): either 'facs' or 'cytof'/'cyTOF'. Defaults to 'facs'.
        copy (bool, optional): Return a copy instead of writing to adata.
            Defaults to False.

    Returns:
        Depending on `copy`, returns or updates `adata` with the following fields:
            AnnData: AnnData object containing area or height data in `.var`
    """
    adata = adata.copy() if copy else adata
    option_key = option
    key_in = key

    possible_options = ["area", "height", "other", "element"]

    if option_key not in possible_options:
        print(f"'{option_key}' is not a valid category. Return all.")
        return adata
    # Check if indices for area and height have been computed
    if key_in not in adata.var_keys():
        adata = find_indexes(adata, var_key=var_key, data_type=data_type)

    index = adata.var[key_in] == option_key
    # if none of the indices is true, abort
    if sum(index) < 1:
        print(f"'{option_key}' is not in adata.var['{key_in}']. Return all.")
        return adata

    non_idx = np.flatnonzero(np.invert(index))

    # merge non-idx entries in data matrix with obs
    non_cols = adata.var_names[non_idx].values
    for idx, colname in enumerate(non_cols):
        adata.obs[colname] = adata.X[:, non_idx[idx]].copy()

    # create new anndata object (note: removes potential objects like obsm)
    adataN = AnnData(
        X=adata.X[:, np.flatnonzero(index)],
        obs=adata.obs,
        var=adata.var.iloc[np.flatnonzero(index)],
        uns=adata.uns,
    )
    adataN.var_names = adata.var_names[index].values
    return adataN if copy else None


# TODO: move function to plotting module
# Plot data. Choose between Area, Height both(default)
def plotdata(
    adata: AnnData,
    key="signal_type",
    normalize=True,
    cofactor=10,
    figsize=(15, 6),
    option="area",
    save="",
    **kwargs,
):
    """Creating histogram plot from Anndata object.

    :param adata: AnnData object containing data.
    :param cofactor: float value to normalize with in arcsinh-transform
    :param option: Switch to choose directly between area and height data.
    :param save: Filename to save the shown figure
    :param kwargs: Passed to :func:`matplotlib.pyplot.savefig`
    """
    option_key = option
    key_in = key
    adata_ = adata.copy()

    # Check if indices for area and height have been computed
    if key_in not in adata_.var_keys():
        find_indexes(adata_)

    if normalize:
        normalize_arcsinh(adata_, cofactor)

    if option_key not in ["area", "height", "other"]:
        print(f"Option {option_key} is not a valid category. Return all.")
        datax = adata_.X
        var_names = adata_.var_names.values
    else:
        index = adata_.var[key_in] == option_key
        datax = adata_.X[:, index]
        var_names = adata_.var_names[index].values

    if len(var_names) == 0:
        print(
            f"Option {option_key} led to the selection of 0 variables.              "
            " Nothing to plot."
        )
        return

    rcParams["figure.figsize"] = figsize

    names = var_names
    number = len(names)

    columns = 3
    rows = math.ceil(number / columns)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.6)

    for idx in range(number):
        ax = fig.add_subplot(rows, columns, idx + 1)
        sb.distplot(
            datax[:, names == names[idx]],
            kde=False,
            norm_hist=False,
            bins=400,
            ax=ax,
            axlabel=names[idx],
        )
    if save != "":
        plt.savefig(save, bbox_inches="tight", **kwargs)
    plt.show()

    return
