#!/usr/bin/env python
from setuptools import setup


if __name__ == '__main__':
    setup(
        name='sample-space',
        version='0.0.0',
        description='Sample Space Builder',
        author='Pierre Delaunay',
        packages=[
            'sspace',
        ],
        setup_requires=['setuptools'],
        tests_require=['pytest', 'flake8', 'codecov', 'pytest-cov'],
    )
