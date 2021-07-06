# setup.py
from setuptools import setup

setup(
    name='latentsafesets',
    version='0.1.0',
    packages=['latentsafesets'],
    install_requires=[
        'pandas',
        'torch',
        'numpy',
        'tqdm',
        'pprintpp',
        'gym',
        'matplotlib',
        'scikit-image',
        'moviepy',
        'Pillow',
        'torchvision',
        'tensorflow==1.15.0',
        'mujoco-py',
    ]
)
