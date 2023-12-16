from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["*.pyx", "KitNET/*.pyx"],
                            compiler_directives={'unraisable_tracebacks': True},
                            annotate=True),
                            include_dirs=[numpy.get_include()]
)
