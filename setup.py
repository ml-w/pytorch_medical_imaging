from setuptools import setup, Extension, find_packages
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
    name="pytorch-medical-imaging",
    version="0.1",  # 與 setup.cfg 保持一致
    packages=find_packages(),  # 自動發現所有套件
    ext_modules=cythonize(extensions, language_level="3"),
)