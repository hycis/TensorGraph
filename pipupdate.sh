#! /usr/bin/bash

version=$1
git tag $version -m "update to version $version"
git push --tag origin master
python setup.py register -r pypi
python setup.py sdist upload -r pypi
