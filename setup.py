from setuptools import setup
from Cython.Build import cythonize
from setuptools import Extension

import numpy as np
_NP_INCLUDE_DIRS = np.get_include()


# Extension modules
ext_modules = [
    Extension(
        name='retinanet.cython_bbox',
        sources=[
            'retinanet/cython_bbox.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    ),
]

setup(
    name='Retinanet',
    packages=['retinanet'],
    ext_modules=cythonize(ext_modules)
)
