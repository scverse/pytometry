from distutils.core import setup
from Cython.Build import cythonize

setup(name='humap',
      ext_modules=cythonize("hmp.pyx"))

