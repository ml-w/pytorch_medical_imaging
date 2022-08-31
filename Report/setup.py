# python setup.py build_ext --inplace

from setuptools import setup, find_packages
from setuptools.extension import Extension

import os

setup(
    name='npc_report_gen',
    version='0.1',
    packages=find_packages(),
    license='',
    author='alabamagan',
    description='',
    entry_points = {
        'console_scripts': [
            'npc_report_gen = npc_report_gen.report_gen_pipeline:main'
        ]
    },
    # scripts = scripts,
    # install_requires=['torchio'],
    # dependency_links=[os.path.abspath('./ThirdParty/torchio')]
)
