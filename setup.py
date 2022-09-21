# python setup.py build_ext --inplace

from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy
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
                  ["pytorch_med_imaging/med_img_dataset/computations/_interpolation.pxd"],
                  include_dirs=[numpy.get_include()]),
    ]
    ext_modules += [
        Extension("pytorch_med_imaging.med_img_dataset.computations._prob_func",
                  ["pytorch_med_imaging/med_img_dataset/computations/_prob_func.pyx"],
                  include_dirs=[numpy.get_include()]),
    ]
    cmdclass.update({'build_ext': build_ext})
    # ext_modules += cythonize("med_img_dataset.rst/computations/_LocalNeighborhoodDifferencePattern.pyx")

import os

scripts = [os.path.join('pytorch_med_imaging/scripts', s) for s in os.listdir('pytorch_med_imaging/scripts')]

setup(
    name='pytorch_medical_imaging',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/alabamagan/pytorch_medical_imaging',
    license='',
    author='ML, Wong',
    author_email='fromosia@link.cuhk.edu.hk',
    description='',
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    entry_points = {
        'console_scripts': [
            'pmi-main = pytorch_med_imaging.main:console_entry',
            'pmi-dicom2nii = pytorch_med_imaging.scripts.dicom2nii:console_entry',
            'pmi-analysis_segment = pytorch_med_imaging.scripts.analysis:segmentation_analysis',
            'pmi-match_dimension = pytorch_med_imaging.scripts.match_dimension:console_entry',
            'pmi-make_masks = pytorch_med_imaging.scripts.make_masks:console_entry',
            'pmi-labels_remap = pytorch_med_imaging.scripts.preprocessing_labelmaps:remap_label',
            'pmi-labels_statistic = pytorch_med_imaging.scripts.preprocessing_labelmaps:pmi_label_statistics',
            'pmi-gen_batch = pytorch_med_imaging.scripts.gen_batch:console_entry'
        ]
    },
    # scripts = scripts,
    # install_requires=['torchio'],
    # dependency_links=[os.path.abspath('./ThirdParty/torchio')]
)
