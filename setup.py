
from setuptools import setup, find_packages

setup(
    name='scout-opt',
    version='0.0.1',
    url='https://github.com/jldaniel/ScoutOpt',
    packages=find_packages(),
    author='Jason Daniel',
    author_email="jdanielae@gmail.com",
    description='Experimental Bayesian Optimization package',
    install_requires=[
        "numpy >= 1.15.0",
        "scipy >= 1.1.0",
        "scikit-learn >= 0.19.2",
    ],
)
