#!/usr/bin/env python
from distutils.core import setup, Extension
import numpy as np

# render README.md as rST
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

setup(name="approxmath",
      version="1.0.1",
      description="Fast approximate math functions: log, exp, sin, cos",
      author="Brendan Ashworth",
      author_email="brendan.ashworth@me.com",
      url="https://github.com/brendanashworth/approxmath",
      long_description=long_description,
      ext_modules=[Extension(
          'approxmath', ['src/approxmath/approxmath.c'],
          extra_compile_args=["-Ofast", "-march=native", "-ffast-math"],
          include_dirs=[np.get_include()])]
)
