#!/usr/bin/env python
from setuptools import setup, Extension, find_packages
import numpy as np

# render README.md as rST
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

setup(name="approxmath",
      version="2.0.0",
      description="Fast approximate math functions: log, exp, sin, cos",
      author="Brendan Ashworth",
      author_email="brendan.ashworth@me.com",
      url="https://github.com/brendanashworth/approxmath",
      long_description=long_description,
      package_dir = {'': 'src'},
      packages=['approxmath/aesara'],
      ext_modules=[Extension(
          'approxmath.np', ['src/approxmath/np/approxmath.c'],
          extra_compile_args=["-Ofast", "-march=native", "-ffast-math"],
          include_dirs=[np.get_include()])]
)
