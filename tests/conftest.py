import functools
import os

import pytest
import readfcs


@pytest.fixture(scope="session", autouse=True)
def download_example_data_once(tmp_path_factory):
    """Make readfcs' example dataset download once per test process, into a directory of its own.

    readfcs downloads it into the working directory under a fixed name on every call, so pytest-xdist workers would
    overwrite each other's copy while reading it.
    """
    directory = tmp_path_factory.mktemp("data")
    download = readfcs.datasets.Oetjen18_t1

    @functools.cache
    def oetjen18_t1() -> str:
        cwd = os.getcwd()
        os.chdir(directory)
        try:
            return str(directory / download())
        finally:
            os.chdir(cwd)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(readfcs.datasets, "Oetjen18_t1", oetjen18_t1)
        yield
