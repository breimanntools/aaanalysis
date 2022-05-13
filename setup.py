#!/usr/bin/env python

from setuptools import setup

if __name__ == "__main__":

    from setuptools import setup

    import sys
    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("cpp_tools requires python >= 3.6.")
    setup(
        name="cpp_tools",
        version='1.0',
        description='Python Distribution of Comparative Physicochemical Profiling (CPP) feature engineering toolkit',
        author='Stephan Breimann',
        author_email='stephanbreimann@yahoo.de',
        url=None,
        packages=['cpp_tools'],
        include_package_data=True,
        package_data={"": ["data/*.xlsx"]}
    )

