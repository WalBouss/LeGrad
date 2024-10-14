""" Setup
Adapted from https://github.com/mlfoundations/open_clip
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

def _read_reqs(relpath):
    fullpath = path.join(path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]

REQUIREMENTS = _read_reqs("requirements.txt")

setup(
    name='legrad_torch',
    version="1.1",
    description='LeGrad',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/WalBouss/LeGrad',
    author='Walid Bousselham, Angie Boggust, Sofian Chaybouti, Hendrik Strobelt, Hilde Kuehne',
    author_email='',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='Explainability, ViT, Vision-Language Models, GradCAM, CLIP pretrained',
    py_modules=["legrad"],
    packages=find_packages(exclude=["assets*"]),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires='>=3.7',
)