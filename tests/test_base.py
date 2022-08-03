import anndata
import readfcs

from pytometry.read_write import read_fcs


def test_read_fcs():
    from urllib.request import urlretrieve

    path_data, _ = urlretrieve(readfcs.datasets.example(), "example.fcs")
    adata = read_fcs(path_data)
    assert isinstance(adata, anndata._core.anndata.AnnData)


# def test_dummy():
#    assert example_function("A") == "a"
#    ex = ExampleClass(1)
#    assert ex.bar() == "hello"
