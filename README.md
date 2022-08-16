[![pypi](https://img.shields.io/pypi/v/pytometry?color=blue&label=pypi%20package)](https://pypi.org/project/pytometry)
[![codecov](https://codecov.io/gh/buettnerlab/pytometry/branch/main/graph/badge.svg?token=AEG5ra92HV)](https://codecov.io/gh/buettnerlab/pytometry)
[![Stars](https://img.shields.io/github/stars/buettnerlab/pytometry?logo=GitHub&color=yellow)](https://github.com/buettnerlab/pytometry/stargazers)
<a href="https://gitmoji.dev">
<img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg" alt="Gitmoji">
</a>

# Pytometry: Flow & mass cytometry analytics

This package provides efficient and scalable handling of flow and mass cytometry data analysis. It provides

- the functionality to read in flow data in the fcs file format as [anndata](https://anndata.readthedocs.io/en/latest/) objects
- flow and mass cytometry specific preprocessing tools
- access to the entire [scanpy](https://scanpy.readthedocs.io/en/stable/) workflow functionality
- GPU support through [rapids](https://github.com/clara-parabricks/rapids-single-cell-examples)

Follow https://twitter.com/marenbuettner to learn about a first public release.

For beta users: Read the [docs](https://pytometry.netlify.app).

You can install `pytometry` via [pip](https://pip.pypa.io/) from [PyPI](PyPI):

```
pip install pytometry
```

or from GitHub:

```
pip install git+https://github.com/buettnerlab/pytometry.git
```
