"""
Created on Mar 15, 2012

@author: Jacob Frelinger
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
from cyarma import include_dir as arma_dir
from cyrand import include_dir as rng_dir

setup(
    name='dpmix_exp',
    version='0.5a',
    packages=['dpmix_exp'],
    package_dir={'dpmix_exp': 'src'},
    description='Optimized (& GPU enhanced) fitting of Gaussian Mixture Models',
    maintainer='Jacob Frelinger',
    maintainer_email='jacob.frelinger@duke.edu',
    author='Andrew Cron',
    author_email='andrew.cron@duke.edu',
    url='https://github.com/andrewcron/pycdp',
    requires=[
        'numpy (>=1.6)',
        'scipy (>=0.6)',
        'matplotlib (>=1.0)',
        'cython (>=0.17)',
        'cyarma (==0.2)',
        'cyrand (>=0.2)',
        'mpi4py'
    ],
    package_data={'dpmix_exp': ['cufiles/*.cu']},
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            "dpmix_exp.munkres",
            ["src/munkres.pyx", "src/cpp/Munkres.cpp"],
            include_dirs=[get_include(), 'src/cpp', 'cpp/'],
            language='c++'
        ),
        Extension(
            "dpmix_exp.sampler",
            ["src/sampler_utils.pyx"],
            include_dirs=[
                get_include(),
                arma_dir,
                rng_dir,
                '/usr/include',
                '/usr/local/include',
                '/opt/local/include'
            ],
            library_dirs=['/usr/lib', '/usr/local/lib', '/opt/local/lib'],
            libraries=['armadillo'],
            language='c++',
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp', '-lgomp']
        )
    ],
)
