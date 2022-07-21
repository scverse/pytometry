"""Flow & mass cytometry analytics.

Import the package::

   import pytometry

This is the complete API reference:

.. autosummary::
   :toctree: .

   example_function
   ExampleClass
"""

__version__ = "0.0.1"  # denote a pre-release for 0.1.0 with 0.1a1

from . import preprocessing as pp
from . import tools as tl
from . import read_write as io
from ._core import ExampleClass, example_function  # noqa
