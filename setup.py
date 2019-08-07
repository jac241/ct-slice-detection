#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

# with open('README.md') as readme_file:
#     readme = readme_file.read()
#
# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = [ 'scipy',
                 'keras',
                 'pandas',
                 'opencv-contrib-python',
                 'numpy',
                 'scikit-learn',
                 'imageio',
                 'imgaug']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Fahdi Kanavati",
    author_email='fahdi.kanavati@imperial.ac.uk',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Code for CT Slice Detection",
    entry_points={
            'console_scripts': [
                'l3_detect_trainer=ct_slice_detection.trainer:main',
                'l3_detect_tester=ct_slice_detection.tester:main'
            ],
        },
    install_requires=requirements,
    license="BSD license",
    long_description=' ',
    include_package_data=True,
    keywords='ct_slice_detection',
    name='ct_slice_detection',
    packages=find_packages(include=['ct_slice_detection']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/fk128/ct_slice_detection',
    version='0.1.0',
    zip_safe=False,
)
