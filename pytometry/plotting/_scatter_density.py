from typing import Literal  # noqa: TYP001
from typing import List, Optional, Tuple, Union

import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from datashader.mpl_ext import dsshow
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.scale import ScaleBase


def scatter_density(
    adata: AnnData,
    x: str = "FSC-A",
    y: str = "SSC-A",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_scale: Union[ScaleBase, Literal["linear", "log", "symlog", "logit"]] = "linear",
    y_scale: Union[ScaleBase, Literal["linear", "log", "symlog", "logit"]] = "linear",
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    cmap: Union[str, List, Colormap] = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    *,
    layer: Optional[str] = None,
):
    """Plots the cell density across two adata.obs.

    Args:
        adata (AnnData): AnnData object containing data.
        x (str): adata.obs to plot on x axis.
            Defaults to 'FSC-A'
        y (str): adata.obs to plot on x axis.
            Defaults to 'SSC-A'.
        x_label (str): x axis label.
        y_label (str): y axis label.
        x_scale (str{"linear", "log", "symlog", "logit", ...}):
            x axis scale type to apply.
            Defaults to 'linear'.
        y_scale (str{"linear", "log", "symlog", "logit", ...}):
            y axis scale type to apply.
            Defaults to 'linear'.
        x_lim (list): upper and lower limit of the x axis.
        y_lim (list): upper and lower limit of the y axis.
        ax (`matplotlib.Axes`), optional:
            Axes to draw into. If *None*, create a new figure or use ``fignum`` to
            draw into an existing figure.
        cmap (str or list or :class:`matplotlib.colors.Colormap`), optional:
            For scalar aggregates, a matplotlib colormap name or instance.
            Alternatively, an iterable of colors can be passed and will be converted
            to a colormap.  Defaults to 'jet'.
        vmin, vmax (float), optional:
            For scalar aggregates, the data range that the colormap covers.
            If vmin or vmax is None (default), the colormap autoscales to the
            range of data in the area displayed, unless the corresponding value is
            already set in the norm.
        layer
            layer in `adata` to use. If `None`, use `adata.X`.

    Returns:
        Scatter plot that displays cell density
    """
    fig, ax = plt.subplots()
    if x_label is None:
        x_label = x
    if y_label is None:
        y_label = y
    # Create df from anndata object
    markers = [x, y]
    joined = sc.get.obs_df(adata, keys=[*markers], layer=layer)

    # Convert variables to np.array
    x = np.array(joined[x])
    y = np.array(joined[y])

    # Plot density with datashader
    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        vmin=vmin,
        vmax=vmax,
        norm=None,
        # aspect="auto",
        ax=ax,
        cmap=cmap,
    )

    plt.colorbar(dsartist)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.yscale(x_scale)
    plt.xscale(y_scale)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()
