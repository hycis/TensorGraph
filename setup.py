from distutils.core import setup
from setuptools import find_packages
import json
from tensorgraph import __version__

setup(
    name='tensorgraph',
    version=__version__,
    author='Joe Wu',
    author_email='hiceen@gmail.com',
    url='https://github.com/hycis/TensorGraph',
    download_url = 'https://github.com/hycis/TensorGraph/tarball/{}'.format(__version__),
    license='Apache 2.0, see LICENCE',
    description='A high level tensorflow library for building deep learning models',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=['numpy>=1.7.1',
                      'six>=1.9.0',
                      'scikit-learn>=0.17',
                      'pandas>=0.17'],
    include_package_data=True,
    zip_safe=False
)
