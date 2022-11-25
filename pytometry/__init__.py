"""Flow & mass cytometry analytics.

Import the package::

   import pytometry

This is the complete API reference:

.. module pytometry.io

.. autosummary::
   :recursive:
   :toctree: .

   io.read_fcs

.. module pytometry.pp

.. autosummary::
   :recursive:
   :toctree: .

   pp.split_signal
   pp.compensate
   pp.find_indexes

..module pytometry.tl

.. autosummary::
   :recursive:
   :toctree: .

   tl.normalize_arcsinh
   tl.normalize_logicle
   tl.normalize_biExp

"""

__version__ = "0.1.3"  # denote a pre-release for 0.1.0 with 0.1a1

from . import plotting as pl
from . import preprocessing as pp
from . import read_write as io
from . import tools as tl
