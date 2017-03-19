from distutils.core import setup
from setuptools import find_packages

version = '3.0.3'
setup(
    name='tensorgraph',
    version=version,
    author='Wu Zhen Zhou',
    author_email='hyciswu@gmail.com',
    install_requires=['numpy>=1.7.1',
                      'six>=1.9.0',
                      'scikit-learn>=0.17',
                      'pandas>=0.17',
                      'scipy>=0.17'],
    url='https://github.com/hycis/TensorGraph',
    download_url = 'https://github.com/hycis/TensorGraph/tarball/{}'.format(version),
    license='Apache 2.0, see LICENCE',
    description='A high level tensorflow library for building models',
    long_description=open('README.md').read(),
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True
)
