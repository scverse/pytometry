from typing import Optional, Tuple

import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib import rcParams

from ..preprocessing._process_data import find_indexes
from ..tools._normalization import normalize_arcsinh, normalize_biExp, normalize_logicle


# Plot data. Choose between Area, Height both(default)
def plotdata(
    adata: AnnData,
    key: str = "signal_type",
    normalize: Optional[str] = None,
    cofactor: float = 10,
    figsize: Tuple[float, float] = (15, 6),
    option: str = "area",
    save: str = "",
    **kwargs,
):
    """Creating histogram plot from Anndata object.

    :param adata: AnnData object containing data.
    :param cofactor: float value to normalize with in arcsinh-transform
    :param normalize: choose between "arcsinh", "biExp" and "logicle"
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

    if normalize is not None:
        if normalize.lower().count("arcsinh") > 0:
            normalize_arcsinh(adata_, cofactor)
        elif normalize.lower().count("biexp") > 0:
            normalize_biExp(adata_)
        elif normalize.lower().count("logicle") > 0:
            normalize_logicle(adata_)
        else:
            print(
                f"{normalize} is not a valid normalization type. Continue without"
                " normalization."
            )

    if option_key.lower() not in ["area", "height", "other"]:
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
    rows = int(np.ceil(number / columns))

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.8, wspace=0.6)

    for idx in range(number):
        ax = fig.add_subplot(rows, columns, idx + 1)
        sns.distplot(
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
