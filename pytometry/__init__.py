"""Flow & mass cytometry analytics.

Import the package::

   import pytometry

This is the complete API reference:

.. autosummary::
   :recursive:
   :toctree: .

   read_write.read_fcs
   preprocessing.split_signal
   preprocessing.compensate
   preprocessing.find_indexes
   tools.normalize_arcsinh
   tools.normalize_logicle
   tools.normalize_biExp

"""

__version__ = "0.1.0"  # denote a pre-release for 0.1.0 with 0.1a1

from . import plotting as pl
from . import preprocessing as pp
from . import read_write as io
from . import tools as tl

# from ._core import ExampleClass, example_function  # noqa
