import flowio
import numpy as np
import pytest

from pytometry.io import read_and_merge


def _write_fcs(path, n_events: int) -> None:
    data = np.random.default_rng(n_events).random((n_events, 3)) * 1000
    with open(path, "wb") as fh:
        flowio.create_fcs(fh, data.ravel().tolist(), ["FSC-A", "SSC-A", "FL1-A"], ["", "", "CD3"])


@pytest.fixture
def fcs_dir(tmp_path):
    _write_fcs(tmp_path / "s2_ctrl.fcs", 30)
    _write_fcs(tmp_path / "s1_stim.FCS", 20)
    (tmp_path / "notes.txt").write_text("not an FCS file")
    return tmp_path


def test_read_and_merge_directory(fcs_dir):
    adata = read_and_merge(str(fcs_dir))
    assert adata.n_obs == 50
    assert adata.obs_names.is_unique
    assert list(adata.var_names) == ["FSC-A", "SSC-A", "CD3"]
    # files are read in name order, s1 (20 cells) before s2 (30 cells)
    assert adata.obs["batch"].value_counts().to_dict() == {"1": 30, "0": 20}
    assert "sample" not in adata.obs


def test_read_and_merge_sample_ids(fcs_dir):
    adata = read_and_merge(str(fcs_dir), sample_ids=["stim", "ctrl"])
    assert adata.obs["sample"].value_counts().to_dict() == {"ctrl": 30, "stim": 20}
    adata = read_and_merge(str(fcs_dir), sample_id_from_filename=True)
    assert adata.obs["sample"].value_counts().to_dict() == {"s2": 30, "s1": 20}
    files = [str(fcs_dir / "s2_ctrl.fcs"), str(fcs_dir / "notes.txt")]
    assert read_and_merge(files, sample_ids=["ctrl"]).obs["sample"].unique().tolist() == ["ctrl"]
    with pytest.raises(ValueError, match="2 FCS files"):
        read_and_merge(str(fcs_dir), sample_ids=["only_one"])
