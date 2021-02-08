from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["wandb==0.10.17",
                     "torch==1.7.1",
                     "ipdb==0.13.4",
                     "numpy==1.19.2",
                     "Bio==0.3.0"]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)