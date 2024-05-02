from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='lefs_cython',
    package_dir={'': 'src'},  # new line to specify the package directory
    packages=find_packages(where='src'),  # find packages in src directory
    ext_modules=cythonize([Extension("lefs_cython.simple", ["src/lefs_cython/simple.pyx"],
                                     include_dirs=[numpy.get_include()])]),
    zip_safe=False,
    install_requires=[
        'numpy',
    ],
)
