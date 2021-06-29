from distutils.core import setup
from setuptools import find_packages
import json
from tensorgraph import __version__

setup(
    name='tensorgraph',
    url='https://github.com/hycis/TensorGraphX',
    download_url = 'https://github.com/hycis/TensorGraphX/tarball/{}'.format(__version__),
    license='Apache 2.0, see LICENCE',
    description='A high level tensorflow library for building deep learning models',
    long_description=open('README.md').read(),
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True
)
