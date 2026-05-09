from setuptools import setup, find_packages

setup(
    name='inknet',
    version='1.0',
    description='Signature feature extraction and writer-dependent verification.',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21',
        'torch>=1.12',
        'torchvision>=0.13',
        'scikit-learn>=1.0',
        'scikit-image>=0.19',
        'tqdm',
    ],
    packages=find_packages(),
)
