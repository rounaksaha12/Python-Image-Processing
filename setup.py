#Rounak Saha
#20CS30043

from setuptools import setup, find_packages

VERSION = '0.0.4' 
DESCRIPTION = 'my_package'
LONG_DESCRIPTION = 'contains modules for Instance Segmentation and Detection, Image transforms, Plot visualization and Dataset creation'

# Setting up
setup(
        name="my_package", 
        version=VERSION,
        author="Rounak Saha",
        author_email="rounaksaha12@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'numpy>=1.14.5',
            'matplotlib>=2.2.0',
            'Pillow',
            'torch',
            'torchvision'
        ]
)