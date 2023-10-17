from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.colors import LogNorm
from matplotlib.widgets import PolygonSelector
from shapely import MultiPoint, Polygon


@dataclass
class PopulationMeta:
    name: str
    parent: str
    x: str
    y: str
    exclude: list[str] | None
    gate_coords: list[tuple[float, float]]
    proportion_of_parent: float
    proportion_of_all: float
    datetime: str

    def __repr__(self):
        """Return a string representation of the object.

        Returns:
            String representation of the object.
        """
        return f"""PopulationMeta(name={self.name}, parent={self.parent})"""

    @classmethod
    def from_dict(cls, data: dict) -> PopulationMeta:
        return PopulationMeta(
            name=data["name"],
            parent=data["parent"],
            x=data["x"],
            y=data["y"],
            exclude=data["exclude"],
            gate_coords=data["gate_coords"],
            proportion_of_parent=data["proportion_of_parent"],
            proportion_of_all=data["proportion_of_all"],
            datetime=data["datetime"],
        )


def make_box_layout():
    return widgets.Layout(
        border="solid 1px black", margin="0px 10px 10px 0px", padding="5px 5px 5px 5px"
    )


def inside_polygon(x: np.ndarray, poly: Polygon) -> np.ndarray:
    """Return elements of x that are inside the polygon.

    Args:
        x: Numpy.Array of shape (n, 2) where each row is a set of coordinates.
        poly: shapely.Polygon object.

    Returns:
        Numpy.Array of shape (m,) where m <= n. Each element is a row index of x that
        is inside the polygon.
    """
    points = MultiPoint(x)
    intersection = poly.intersection(points)
    if intersection.geom_type == "MultiPoint":
        coords = np.array(intersection.coords)
    elif intersection.geom_type == "Point":
        coords = np.array([intersection.coords[0]])
    else:
        return np.array([])
    return np.where(np.isin(x, coords).all(axis=1))[0]


class Gate(widgets.HBox):
    def __init__(
        self,
        adata: AnnData,
        x: str,
        y: str,
        obs_name: str = "population",
        parent: str = "root",
        exclude: list[str] | None = None,
        cmap: str = "jet",
        bins: int | None = 250,
        figsize: tuple[int, int] = (5, 5),
        xaxis_limits: tuple[float, float] | None = None,
        yaxis_limits: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.obs_name = obs_name
        self.parent = parent
        self.exclude = exclude or []
        self.bins = bins
        self.cmap = cmap
        self.xaxis_limits = xaxis_limits
        self.yaxis_limits = yaxis_limits

        if exclude:
            self._plot_data = np.array(
                adata[
                    ~adata.obs[obs_name].isin(exclude) & (adata.obs[obs_name] == parent)
                ][:, [x, y]].X
            )
        else:
            self._plot_data = np.array(
                adata[adata.obs[obs_name] == parent][:, [x, y]].X
            )

        self._data = adata.copy()
        self._new_pop_idx = None

        # Define canvas
        output = widgets.Output()
        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=figsize)
        self.fig.canvas.toolbar_position = "bottom"

        # Define widgets
        self.selector = PolygonSelector(self.ax, lambda i: None)
        self.apply_button = widgets.Button(description="Apply Gate")
        self.apply_button.on_click(self._apply_click)
        self.new_pop_size_parent = widgets.Text(value="")
        self.new_pop_size_all = widgets.Text(value="")

        controls = widgets.VBox([self.apply_button])
        text_outputs = widgets.VBox([self.new_pop_size_parent, self.new_pop_size_all])
        controls.layout = make_box_layout()
        _ = widgets.Box([output])
        output.layout = make_box_layout()
        self.children = [controls, text_outputs, output]

        self._plot()

    def apply(self, new_population_name: str, meta_log: Path | str) -> AnnData:
        meta_log = Path(meta_log)
        if self._new_pop_idx is None:
            raise ValueError("No new population was selected.")
        if self.obs_name not in self._data.obs.columns:
            self._data.obs[self.obs_name] = "root"
        self._data.obs.loc[self._new_pop_idx, self.obs_name] = new_population_name

        if not meta_log.exists():
            meta_log.mkdir(parents=True, exist_ok=True)
            meta_data = []
        else:
            with open(meta_log, "r") as f:
                meta_data = json.load(f)
        meta_data.append(
            asdict(
                PopulationMeta(
                    name=new_population_name,
                    x=self.x,
                    y=self.y,
                    parent=self.parent,
                    exclude=self.exclude,
                    gate_coords=self.selector.verts,
                    proportion_of_parent=self.new_pop_size_parent.value,
                    proportion_of_all=self.new_pop_size_all.value,
                    datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
        )
        with open(meta_log, "w") as f:
            json.dump(meta_data, f, indent=4)
        return self._data

    def _plot(self):
        bins = self.bins or int(np.sqrt(self._plot_data.shape[0]))
        self.ax.hist2d(
            self._plot_data[:, 0],
            self._plot_data[:, 1],
            bins=[bins, bins],
            cmap=self.cmap,
            norm=LogNorm(),
        )

        if self.xaxis_limits:
            self.ax.set_xlim(self.xaxis_limits)
        if self.yaxis_limits:
            self.ax.set_ylim(self.yaxis_limits)

        self.ax.set_title(f"Parent:{self.parent}; excluding:{self.exclude}")
        self.ax.set_xlabel(self.x)
        self.ax.set_ylabel(self.y)

    def _apply_click(self):
        verts = self.selector.verts
        verts.append(verts[0])
        polygon = Polygon(verts)
        self._new_pop_idx = inside_polygon(self._plot_data, polygon)
        parent_data = self._data[self._data.obs[self.obs_name] == self.parent]
        prop_of_parent = len(self._new_pop_idx) / parent_data.shape[0]
        prop_of_all = len(self._new_pop_idx) / self._data.shape[0]
        self.new_pop_size_parent.value = (
            f"Population size (% of parent): {prop_of_parent}"
        )
        self.new_pop_size_all.value = f"Population size (% of all): {prop_of_all}"
