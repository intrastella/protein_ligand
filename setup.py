from setuptools import find_packages
from setuptools import setup


setup(
    name="ligand_prediction",
    use_scm_version=True,
    packages=find_packages(),
    classifiers=[
            # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Operating System :: Unix',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7'
        ],

    include_package_data=True,
    author="Stella Muamba Ngufulu",
    author_email="stellamuambangufulu@gmail.com",
    description="Ligand prediction repo for experimental purposes.",
    license="MIT License",
    keywords="ligand prediction, dataloader processing, library",
    platforms="linux")
