import os
import numpy

from sklearn._build_utils import gen_from_templates


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("tools", parent_package, top_path)

    libraries = []
    if os.name == "posix":
        libraries.append("m")


    # generate humap_fast from template
    templates = ["pytometry/tools/_sag_fast.pyx.tp"]
    gen_from_templates(templates)

    config.add_extension(
        "_sag_fast", sources=["_sag_fast.pyx"], include_dirs=numpy.get_include()
    )

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
