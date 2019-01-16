from setuptools import setup
from setuptools.extension import Extension

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
        Extension("MedImgDataset.Computation._LocalNeighborhoodDifferencePattern",
                  ["MedImgDataset/Computation/_LocalNeighborhoodDifferencePattern.pyx"]),
    ]
    ext_modules += [
        Extension("MedImgDataset.Computation._interpolation",
                  ["MedImgDataset/Computation/_interpolation.pxd"]),
    ]
    cmdclass.update({'build_ext': build_ext})
    # ext_modules += cythonize("MedImgDataset/Computation/_LocalNeighborhoodDifferencePattern.pyx")

setup(
    name='NPC_Segment',
    version='0.1',
    packages=['Loss', 'Networks', 'Networks.Layers', 'Algorithms', 'MedImgDataset', 'MedImgDataset.Computation'],
    url='https://github.com/teracamo/pytorch_medical_imaging/tree/NPC_Segment',
    license='',
    author='Wong Matthew Lun',
    author_email='fromosia@link.cuhk.edu.hk',
    description='',
    cmdclass = cmdclass,
    ext_modules = ext_modules
)
