from distutils.core import setup
from setuptools import find_packages
import json
<<<<<<< HEAD
from tensorgraph import __version__

setup(
    name='tensorgraph',
=======
from tensorgraphx import __version__

setup(
    name='tensorgraphx',
>>>>>>> e55a706e1467da7b7c54b6d04055aba847f5a2b5
    version=__version__,
    author='Wu Zhen Zhou',
    author_email='hyciswu@gmail.com',
    install_requires=['numpy>=1.7.1',
                      'six>=1.9.0',
                      'scikit-learn>=0.17',
                      'pandas>=0.17',
                      'scipy>=0.17'],
<<<<<<< HEAD
    url='https://skymed.ai/AI-Platform/TensorGraph',
    download_url = 'https://skymed.ai/AI-Platform/TensorGraph/tarball/{}'.format(__version__),
=======
    url='https://github.com/hycis/TensorGraphX',
    download_url = 'https://github.com/hycis/TensorGraphX/tarball/{}'.format(__version__),
>>>>>>> e55a706e1467da7b7c54b6d04055aba847f5a2b5
    license='Apache 2.0, see LICENCE',
    description='A high level tensorflow library for building deep learning models',
    long_description=open('README.md').read(),
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True
)
