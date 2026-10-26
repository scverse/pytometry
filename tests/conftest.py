import os

import pytest
import readfcs


@pytest.fixture(scope="session")
def oetjen18_t1(tmp_path_factory) -> str:
    """Path to the Oetjen18 T cell panel FCS file, downloaded once per test process.

    readfcs downloads it into the working directory under a fixed name on every call, so pytest-xdist workers would
    overwrite each other's copy while reading it.
    """
    directory = tmp_path_factory.mktemp("data")
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        return str(directory / readfcs.datasets.Oetjen18_t1())
    finally:
        os.chdir(cwd)
