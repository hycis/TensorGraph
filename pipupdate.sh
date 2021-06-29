#! /usr/bin/bash

version=$1
git tag $version -m "update to version $version"
<<<<<<< HEAD
git push --tag

# python setup.py register -r pypi
# python setup.py sdist upload -r pypi
=======
git push --tag origin master
python setup.py register -r pypi
python setup.py sdist upload -r pypi
>>>>>>> e55a706e1467da7b7c54b6d04055aba847f5a2b5
