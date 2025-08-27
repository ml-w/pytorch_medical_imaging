from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "pytorch_med_imaging.pmi_data.computations._LocalNeighborhoodDifferencePattern",
                  ["pytorch_med_imaging/pmi_data/computations/_LocalNeighborhoodDifferencePattern.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "pytorch_med_imaging.pmi_data.computations._interpolation",
                  ["pytorch_med_imaging/pmi_data/computations/_interpolation.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "pytorch_med_imaging.pmi_data.computations._prob_func",
                  ["pytorch_med_imaging/pmi_data/computations/_prob_func.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]
setup(
    name="pytorch-med-imaging",
    version="1.0",
    packages=["pytorch_med_imaging.pmi_data.computations"],
    ext_modules=cythonize(extensions, language_level="3"),
)

