# python setup.py build_ext --inplace
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True


cmdclass = {}
ext_modules = []
if use_cython:
    ext_modules += [
        Extension("pytorch_med_imaging.med_img_dataset.computations._LocalNeighborhoodDifferencePattern",
                  ["pytorch_med_imaging/med_img_dataset/computations/_LocalNeighborhoodDifferencePattern.pyx"],
                  include_dirs=[numpy.get_include()]),
    ]
    ext_modules += [
        Extension("pytorch_med_imaging.med_img_dataset.computations._interpolation",
                  ["pytorch_med_imaging/med_img_dataset/computations/_interpolation.pyx"],
                  include_dirs=[numpy.get_include()]),
    ]
    ext_modules += [
        Extension("pytorch_med_imaging.med_img_dataset.computations._prob_func",
                  ["pytorch_med_imaging/med_img_dataset/computations/_prob_func.pyx"],
                  include_dirs=[numpy.get_include()]),
    ]
    cmdclass.update({'build_ext': build_ext})

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass
)
