from typing import Optional  # Special
from typing import Tuple  # Classes

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
    option: str = "area",
    normalize: Optional[str] = None,
    cofactor: Optional[float] = 10,
    figsize: Tuple[float, float] = (15, 6),
    bins: int = 400,
    save: Optional[str] = None,
    n_cols: int = 3,
    **kwargs,
):
    """Creating histogram plot of channels from Anndata object.

    Args:
        adata (AnnData): Anndata object containing data.
        key (str):
            Key in adata.var to plot. Default is 'signal_type' which is generated
            when calling the preprocessing function `split_signal`.
        normalize (str):
            Normalization type. Default is None but can be set to "arcsinh", "biExp"
            or "logicle"
        cofactor (float):
            Cofactor for arcsinh normalization. Default is 10.
        figsize (tuple):
            Figure size (width, height). Default is (15, 6).
        option (str):
            Switch to choose directly between area and height data. Default is "area".
        bins (int):
            Number of bins for the histogram. Default is 400.
        save (str, optional):
            Path to save the figure.
        **kwargs:
            Additional arguments passed to `matplotlib.pyplot.savefig`

    Returns:
    matplotlib.pyplot.Figure
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

    columns = n_cols
    rows = int(np.ceil(number / columns))

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.8, wspace=0.6)

    for idx in range(number):
        ax = fig.add_subplot(rows, columns, idx + 1)
        sns.histplot(datax[:, names == names[idx]], bins=bins, ax=ax, legend=False)
        ax.set_xlabel(names[idx])
    if save:
        plt.savefig(save, bbox_inches="tight", **kwargs)
    return fig
