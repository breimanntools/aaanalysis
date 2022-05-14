#!/usr/bin/env python

from setuptools import setup

if __name__ == "__main__":

    from setuptools import setup

    import sys
    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("AAanalysis requires python >= 3.6.")
    setup(
        name='AAanalysis',
        version='1.0',
        description='Python Distribution of AAanalysis toolkit',
        author='Stephan Breimann',
        author_email='stephanbreimann@gmail.com',
        url=None,
        # Include sub-packages!
        packages=['aaanalysis', 'aaanalysis.aaclust', 'aaanalysis.cpp'],
        #include_package_data=True,
        #package_data={"": ["data/*.xlsx"]}
    )

