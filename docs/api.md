# API

Import pytometry as

```python
import pytometry as pt
```

## Read/write (`io`)

```{eval-rst}
.. module:: pytometry.io
.. currentmodule:: pytometry

.. autosummary::
    :toctree: generated

    io.read_fcs
    io.read_and_merge
```

## Preprocessing (`pp`)

```{eval-rst}
.. module:: pytometry.pp
.. currentmodule:: pytometry

.. autosummary::
   :toctree: generated

   pp.split_signal
   pp.compensate
   pp.create_comp_mat
   pp.find_indexes
```

## Tools

```{eval-rst}
.. module:: pytometry.tl
.. currentmodule:: pytometry

.. autosummary::
   :toctree: generated

   tl.normalize_arcsinh
   tl.normalize_logicle
   tl.normalize_biexp
   tl.normalize_autologicle
```

## Plotting

```{eval-rst}
.. module:: pytometry.pl
.. currentmodule:: pytometry

.. autosummary::
    :toctree: generated

    pl.plotdata
    pl.scatter_density
```
