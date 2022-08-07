import anndata
import readfcs

from pytometry.read_write import read_fcs


def test_read_fcs():
    path_data = readfcs.datasets.example()
    adata = read_fcs(path_data)
    assert isinstance(adata, anndata._core.anndata.AnnData)
