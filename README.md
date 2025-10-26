# pytometry: Flow & mass cytometry analytics

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/pytometry/test.yaml?branch=main
[link-tests]: https://github.com/scverse/pytometry/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/pytometry

This package provides efficient and scalable handling of flow and mass cytometry data analysis. It provides

- the functionality to read in flow data in the fcs file format as [anndata](https://anndata.readthedocs.io/en/latest/) objects
- flow and mass cytometry specific preprocessing tools
- access to the entire [scanpy](https://scanpy.readthedocs.io/en/stable/) workflow functionality
- GPU support through [rapids-singlecell](https://rapids-singlecell.readthedocs.io/)

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

- [API documentation][link-api].

## Installation

You need to have Python 3.11 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install pytometry:

1. Install the latest release of `pytometry` from [PyPI][link-pypi].

```bash
pip install pytometry
```

2. Install the latest development version:

```bash
pip install git+https://github.com/scverse/pytometry.git@main
```

3. Install locally for development:

Please refer to the [Contributing Guide][contributing].

## Release notes

See [GitHub releases][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

Pytometry is currently a pre-print on [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.10.511546v1).

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/pytometry/issues
[contributing]: https://pytometry.readthedocs.io/en/stable/contributing.html
[changelog]: https://github.com/scverse/pytometry/releases
[link-docs]: https://pytometry.readthedocs.io
[link-api]: https://pytometry.readthedocs.io/latest/api.html
[link-pypi]: https://pypi.org/project/pytometry
[//]: # "numfocus-fiscal-sponsor-attribution"

pytometry is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>
