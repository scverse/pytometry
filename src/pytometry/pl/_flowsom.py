from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from natsort import natsorted

from ._som_utils import (
    _category_colors,
    _doc_params,
    _draw_outlines,
    _draw_wedges,
    _flowsom_result,
    _legend,
    _markers,
    _som_axes,
    doc_ax,
    doc_som_plot,
)


@_doc_params(som_plot=doc_som_plot, ax=doc_ax)
def flowsom_stars(
    adata: AnnData,
    key: str = "clusters",
    markers: Sequence[str] | None = None,
    layout: Literal["mst", "grid"] = "mst",
    node_size: Literal["counts"] | None = "counts",
    max_node_size: float = 0.45,
    show_meta_clusters: bool = True,
    cmap: str | Colormap = "turbo",
    legend: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Axes:
    """Plot every SOM node as a star chart of its marker values, as FlowSOM's `PlotStars`.

    Each node shows one wedge per marker, whose length is the node's weight for that marker, scaled to [0, 1] over
    all nodes; its background is colored by the node's meta-cluster.

    Parameters
    ----------
    adata
        AnnData object clustered with :func:`pytometry.tl.flowsom_clustering`.
    {som_plot}
    markers
        markers to show, by default all markers the SOM was trained on.
    show_meta_clusters
        whether to color the background of every node by its meta-cluster.
    cmap
        colormap from which the markers' colors are taken.
    legend
        whether to show legends of the markers and meta-clusters.
    {ax}

    Returns
    -------
    the Axes drawn into
    """
    result = _flowsom_result(adata, key)
    markers, values = _markers(result, markers)
    ax, positions, radii = _som_axes(adata, key, layout, node_size, max_node_size, ax, figsize)
    if show_meta_clusters:
        labels = [f"cluster_{c}" for c in np.asarray(result["meta_class"]).ravel()]
        meta_colors = _category_colors(adata, key, natsorted(set(labels)))
        backgrounds = PatchCollection(
            [Circle(p, 1.2 * max_node_size) for p in positions],
            facecolors=[meta_colors[label] for label in labels],
            alpha=0.35,
            zorder=1,
        )
        ax.add_collection(backgrounds)
    span = np.ptp(values, axis=0)
    scaled = (values - values.min(axis=0)) / np.where(span > 0, span, 1)
    marker_colors = colormaps.get_cmap(cmap)(np.linspace(0.05, 0.95, len(markers)))
    angles = np.broadcast_to(np.linspace(0, 360, len(markers) + 1), (len(positions), len(markers) + 1))
    _draw_wedges(ax, positions, radii[:, None] * scaled, angles, np.broadcast_to(marker_colors, (*scaled.shape, 4)))
    _draw_outlines(ax, positions, radii)
    if legend:
        marker_legend = _legend(
            ax, dict(zip(markers, marker_colors, strict=True)), "markers", ncols=1 + len(markers) // 25
        )
        if show_meta_clusters:
            ax.add_artist(marker_legend)
            marker_legend.set_clip_on(False)  # add_artist clips it to the axes
            _legend(ax, meta_colors, "meta-clusters", right=False, alpha=0.35)
    return ax


@_doc_params(som_plot=doc_som_plot)
def flowsom_markers(
    adata: AnnData,
    markers: Sequence[str] | None = None,
    key: str = "clusters",
    layout: Literal["mst", "grid"] = "mst",
    node_size: Literal["counts"] | None = "counts",
    max_node_size: float = 0.45,
    cmap: str | Colormap = "viridis",
    n_cols: int = 4,
    panel_size: float = 4,
) -> Figure:
    """Plot the SOM nodes colored by their weight for each marker, one panel per marker, as FlowSOM's `PlotMarker`.

    Parameters
    ----------
    adata
        AnnData object clustered with :func:`pytometry.tl.flowsom_clustering`.
    markers
        markers to show, by default all markers the SOM was trained on.
    {som_plot}
    cmap
        colormap of the marker values.
    n_cols
        number of panels per row.
    panel_size
        width and height of each panel.

    Returns
    -------
    the figure
    """
    markers, values = _markers(_flowsom_result(adata, key), markers)
    n_cols = min(n_cols, len(markers))
    n_rows = -(-len(markers) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False)
    for ax in axes.ravel()[len(markers) :]:
        ax.set_axis_off()
    for ax, marker, marker_values in zip(axes.ravel()[: len(markers)], markers, values.T, strict=True):
        _, positions, radii = _som_axes(adata, key, layout, node_size, max_node_size, ax, None)
        nodes = PatchCollection(
            [Circle(p, r) for p, r in zip(positions, radii, strict=True)],
            cmap=cmap,
            norm=Normalize(marker_values.min(), marker_values.max()),
            edgecolors="0.4",
            linewidths=0.5,
            zorder=2,
        )
        nodes.set_array(marker_values)
        ax.add_collection(nodes)
        fig.colorbar(nodes, ax=ax, shrink=0.6)
        ax.set_title(marker)
    return fig


@_doc_params(som_plot=doc_som_plot, ax=doc_ax)
def flowsom_pies(
    adata: AnnData,
    groupby: str,
    key: str = "clusters",
    layout: Literal["mst", "grid"] = "mst",
    node_size: Literal["counts"] | None = "counts",
    max_node_size: float = 0.45,
    legend: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Axes:
    """Plot every SOM node as a pie chart of the categories of its cells, as FlowSOM's `PlotPies`.

    Useful to compare the SOM with manual gates or to see how the cells of samples or conditions spread over it.

    Parameters
    ----------
    adata
        AnnData object clustered with :func:`pytometry.tl.flowsom_clustering`.
    groupby
        categorical column of `adata.obs`, e.g. manual gates, samples or conditions.
    {som_plot}
    legend
        whether to show a legend of the categories.
    {ax}

    Returns
    -------
    the Axes drawn into
    """
    ax, positions, radii = _som_axes(adata, key, layout, node_size, max_node_size, ax, figsize)
    colors = _category_colors(adata, groupby)
    counts = pd.crosstab(adata.obs[f"{key}_som_node"].to_numpy(), adata.obs[groupby].to_numpy())
    counts = counts.reindex(index=range(len(positions)), columns=list(colors), fill_value=0).to_numpy()
    fractions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    angles = 90 + 360 * np.column_stack([np.zeros(len(positions)), np.cumsum(fractions, axis=1)])
    slice_colors = np.broadcast_to(np.array(list(colors.values())), (*fractions.shape, 4))
    _draw_wedges(ax, positions, np.broadcast_to(radii[:, None], fractions.shape), angles, slice_colors)
    _draw_outlines(ax, positions, radii)
    if legend:
        _legend(ax, colors, groupby)
    return ax
