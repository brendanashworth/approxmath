#!/usr/bin/env python
from distutils.core import setup, Extension
import numpy as np

setup(name="approxmath",
      version="1.0.0",
      description="Fast approximate math functions: log, exp, sin, cos",
      author="Brendan Ashworth",
      author_email="brendan.ashworth@me.com",
      ext_modules=[Extension(
          'approxmath', ['src/approxmath/approxmath.c'],
          extra_compile_args=["-Ofast", "-march=native", "-ffast-math"],
          include_dirs=[np.get_include()])]
)
