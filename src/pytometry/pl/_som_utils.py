from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Patch, Wedge
from natsort import natsorted

doc_som_plot = """\
key
        `key_added` of :func:`pytometry.tl.flowsom_clustering`.
    layout
        "mst" places the nodes along the minimum spanning tree of their weights, as FlowSOM does; "grid" places
        them on the grid of the SOM (not for k-means centroids). The tree only serves to lay the nodes out in two
        dimensions. It does not show a lineage or a development over time, and nodes or meta-clusters drawn next to
        each other are not necessarily more similar to each other than to the rest.
    node_size
        "counts" scales the area of each node with its number of cells; None draws all nodes the same size.
    max_node_size
        radius of the largest node, in units of the mean edge length (MST) or of the grid spacing."""

doc_ax = """\
ax
        Axes to draw into. If None, a new figure is created.
    figsize
        Figure size (width, height) if ax is not provided."""


def _doc_params(**replacements: str) -> Callable:
    def decorate(func: Callable) -> Callable:
        func.__doc__ = func.__doc__.format_map(replacements)
        return func

    return decorate


def _flowsom_result(adata: AnnData, key: str) -> dict:
    if key not in adata.uns or "weights" not in adata.uns[key] or f"{key}_som_node" not in adata.obs:
        raise KeyError(
            f"No FlowSOM result under {key!r}, run pytometry.tl.flowsom_clustering(adata, key_added={key!r})"
        )
    return adata.uns[key]


def _markers(result: dict, markers: Sequence[str] | None) -> tuple[list[str], np.ndarray]:
    """The markers and their columns of the SOM weights."""
    var_names = list(result["var_names"])
    markers = var_names if markers is None else list(markers)
    missing = [m for m in markers if m not in var_names]
    if missing:
        raise KeyError(f"Markers not in the SOM: {missing}")
    return markers, np.asarray(result["weights"])[:, [var_names.index(m) for m in markers]]


def _layout(adata: AnnData, key: str, layout: Literal["mst", "grid"]) -> tuple[np.ndarray, np.ndarray]:
    """Node positions, with nodes joined by an edge on average one unit apart, and the edges to draw."""
    result = adata.uns[key]
    weights = np.asarray(result["weights"])
    no_edges = np.empty((0, 2), dtype=np.int64)
    grid = None
    if result["params"]["method"] == "som":
        n_rows, n_cols = result["params"]["som_dim"]
        rows, cols = np.divmod(np.arange(n_rows * n_cols), n_cols)
        grid = np.column_stack([cols, -rows]).astype(float)
    if layout == "grid":
        if grid is None:
            raise ValueError('k-means centroids have no grid, use layout="mst"')
        return grid, no_edges
    if layout != "mst":
        raise ValueError('Unknown layout, must be one of "mst" or "grid"')
    if "mst_layout" in result:
        return np.asarray(result["mst_layout"]["positions"]), np.asarray(result["mst_layout"]["edges"])
    if len(weights) == 1:
        return np.zeros((1, 2)), no_edges
    from fastersom import minimum_spanning_tree  # imports torch, so only when needed

    tree = minimum_spanning_tree(weights, result["params"]["metric"], device="cpu")
    # Kamada-Kawai, started from a SOM's grid, places the nodes at their path lengths in the tree, as FlowSOM does;
    # identical nodes, joined by edges of length 0, get a small length instead
    floor = 1e-3 * max(tree.weights.max(), 1e-12)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(weights)))
    graph.add_weighted_edges_from(
        (int(a), int(b), max(float(w), floor)) for (a, b), w in zip(tree.edges, tree.weights, strict=True)
    )
    found = nx.kamada_kawai_layout(graph, pos=None if grid is None else dict(enumerate(grid)), weight="weight")
    positions = np.array([found[node] for node in range(len(weights))])
    edges = np.asarray(tree.edges)
    positions /= np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1).mean()
    # Kamada-Kawai is quadratic in the number of nodes, so the layout is kept with the clustering; not in a view,
    # which anndata would copy on writing
    if not adata.is_view:
        result["mst_layout"] = {"positions": positions, "edges": edges}
    return positions, edges


def _node_radii(adata: AnnData, key: str, n_nodes: int, node_size: Literal["counts"] | None, max_size: float):
    """Radii proportional to the square root of the number of cells per node, so that areas follow the counts."""
    if node_size is None:
        return np.full(n_nodes, max_size)
    nodes = adata.obs[f"{key}_som_node"].to_numpy()
    counts = np.bincount(nodes[nodes >= 0], minlength=n_nodes)
    return max_size * np.sqrt(counts / max(counts.max(), 1)).clip(0.1, None)


def _category_colors(adata: AnnData, key: str, categories: list[str] | None = None) -> dict[str, tuple]:
    """Colors of the categories of adata.obs[key], or of the given ones: scanpy's if it set them for these."""
    column = adata.obs[key]
    observed = list(column.cat.categories) if isinstance(column.dtype, pd.CategoricalDtype) else []
    categories = categories or observed or natsorted(set(column.dropna()))
    colors = adata.uns.get(f"{key}_colors", [])
    if observed == categories and len(colors) == len(categories):
        return {c: to_rgba(color) for c, color in zip(categories, colors, strict=True)}
    palette = _default_palette(len(categories))
    return {c: to_rgba(palette[i % len(palette)]) for i, c in enumerate(categories)}


def _default_palette(n_colors: int) -> list:
    """Default palette for categories: scanpy's if it is installed, else matplotlib's tab20, tab20b and tab20c."""
    try:
        from scanpy.plotting.palettes import default_20, default_28, default_102
    except ImportError:
        return [*colormaps["tab20"].colors, *colormaps["tab20b"].colors, *colormaps["tab20c"].colors]
    return default_20 if n_colors <= 20 else default_28 if n_colors <= 28 else default_102


def _som_axes(
    adata: AnnData,
    key: str,
    layout: Literal["mst", "grid"],
    node_size: Literal["counts"] | None,
    max_node_size: float,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> tuple[Axes, np.ndarray, np.ndarray]:
    """The axes, with the tree drawn and the limits set, and the positions and radii of the nodes."""
    _flowsom_result(adata, key)
    positions, edges = _layout(adata, key, layout)
    radii = _node_radii(adata, key, len(positions), node_size, max_node_size)
    ax = plt.subplots(figsize=figsize or (8, 8))[1] if ax is None else ax
    ax.add_collection(LineCollection(positions[edges], colors="0.6", linewidths=1, zorder=0))
    margin = 1.3 * max_node_size
    ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax, positions, radii


def _draw_wedges(ax: Axes, positions: np.ndarray, radii: np.ndarray, angles: np.ndarray, colors: np.ndarray) -> None:
    """Draw one wedge per node and slice, skipping empty ones.

    radii are of shape (n_nodes, n_slices), colors (n_nodes, n_slices, 4) and angles, in degrees, (n_nodes, n_slices + 1).
    """
    centres = np.repeat(positions, radii.shape[1], axis=0)
    radii, starts, ends = radii.ravel(), angles[:, :-1].ravel(), angles[:, 1:].ravel()
    drawn = (radii > 0) & (ends > starts)
    wedges = [
        Wedge(c, r, a, b) for c, r, a, b in zip(centres[drawn], radii[drawn], starts[drawn], ends[drawn], strict=True)
    ]
    ax.add_collection(PatchCollection(wedges, facecolors=colors.reshape(-1, 4)[drawn], edgecolors="none", zorder=2))


def _draw_outlines(ax: Axes, positions: np.ndarray, radii: np.ndarray) -> None:
    circles = [Circle(p, r) for p, r in zip(positions, radii, strict=True)]
    ax.add_collection(PatchCollection(circles, facecolors="none", edgecolors="0.4", linewidths=0.5, zorder=3))


def _legend(ax: Axes, colors: dict, title: str, right: bool = True, alpha: float = 1.0, ncols: int = 1):
    return ax.legend(
        handles=[Patch(color=c, alpha=alpha, label=label) for label, c in colors.items()],
        title=title,
        loc="upper left" if right else "upper right",
        bbox_to_anchor=(1, 1) if right else (0, 1),
        frameon=False,
        ncols=ncols,
    )
