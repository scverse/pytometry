"""Flow & mass cytometry analytics.

Import the package::

   import pytometry

This is the complete API reference:

Read/write (`io`)
=================

.. autosummary::
   :recursive:
   :toctree: .

   read_write.read_fcs

Preprocessing (`pp`)
====================

.. autosummary::
   :recursive:
   :toctree: .

   preprocessing.split_signal
   preprocessing.compensate
   preprocessing.create_comp_mat
   preprocessing.find_indexes

Tools (`tl`)
============

.. autosummary::
   :recursive:
   :toctree: .

   tools.normalize_arcsinh
   tools.normalize_logicle
   tools.normalize_biExp

Plotting (`pl`)
===============

.. autosummary::
   :recursive:
   :toctree: .

   plotting.plotdata
   plotting.scatter_density

"""

__version__ = "0.1.4"  # denote a pre-release for 0.1.0 with 0.1a1

from . import plotting as pl
from . import preprocessing as pp
from . import read_write as io
from . import tools as tl
