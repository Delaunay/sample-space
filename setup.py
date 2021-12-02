#!/usr/bin/env python
from setuptools import setup


if __name__ == '__main__':
    setup(
        name='sspace',
        version='0.0.0',
        description='Sample Space Builder',
        author='Pierre Delaunay',
        packages=[
            'sspace',
            'sspace.backends'
        ],
        setup_requires=['setuptools'],
        install_requires=[
            'ConfigSpace',
            'orion',
        ]
    )
