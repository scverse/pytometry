from typing import Literal  # noqa: TYP001
from typing import List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
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
    figsize: Optional[tuple[int, int]] = None,
    bins: Union[int, tuple[int, int]] = 500,
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
        figsize (tuple), optional:
            Figure size (width, height) if ``ax`` not provided. Defaults to (10, 10).
        bins (int or tuple), optional:
            Number of bins for the `np.histogram2d` function
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
            The layer in `adata` to use. If `None`, use `adata.X`.

    Returns:
        Scatter plot that displays cell density
    """
    ax = plt.subplots(figsize=figsize)[1] if ax is None else ax

    if isinstance(bins, int):
        bins = (bins, bins)

    hist, xedges, yedges = np.histogram2d(
        adata.obs_vector(x, layer=layer), adata.obs_vector(y, layer=layer), bins=bins
    )

    vmin = hist.min() if vmin is None else vmin
    vmax = hist.max() if vmax is None else vmax

    image = ax.imshow(
        hist.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        cmap=_get_cmap_white_background(cmap),
        aspect="auto",
        origin="lower",
    )
    plt.colorbar(image, ax=ax)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_yscale(x_scale)
    ax.set_xscale(y_scale)
    ax.set_xlabel(x if x_label is None else x_label)
    ax.set_ylabel(y if y_label is None else y_label)

    plt.show()


def _get_cmap_white_background(cmap: Union[str, List, Colormap]) -> Colormap:
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    colors = cmap(np.arange(cmap.N))
    colors[0] = np.array([1, 1, 1, 1])

    return mcolors.ListedColormap(colors)
