#!/usr/bin/env python
from distutils.core import setup, Extension
import numpy as np

# render README.md on pypi
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name="approxmath",
      version="1.0.1",
      description="Fast approximate math functions: log, exp, sin, cos",
      author="Brendan Ashworth",
      author_email="brendan.ashworth@me.com",
      long_description=long_description,
      long_description_content_type='text/markdown',
      ext_modules=[Extension(
          'approxmath', ['src/approxmath/approxmath.c'],
          extra_compile_args=["-Ofast", "-march=native", "-ffast-math"],
          include_dirs=[np.get_include()])]
)
