from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

'''
Uncoment '-fopenmp' to compile parallel version
'''

examples_extension = Extension(
    name="pysparsematrixdot",
    sources=["pysparsematrixdot.pyx", "lib/sparsematrixdot.c"],
    libraries=["sparsematrixdot"],
    library_dirs=["lib"],
    include_dirs=["lib", numpy.get_include()],
    #extra_compile_args=['-fopenmp'],
    #extra_link_args=['-fopenmp'],
    language='c'
)

setup(
    name="sparsematrixdot",
    ext_modules=cythonize([examples_extension])
)
