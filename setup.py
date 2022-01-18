# Author: Silas João Bezerra Soares.
# License: MIT

from setuptools import setup

DISTNAME = 'Phase Recovery'
DESCRIPTION = 'Optical Communications Algorithms with Python'
LONG_DESCRIPTION = open('README.md', encoding = "utf8").read()
MAINTAINER = 'Silas João Bezerra Soares'
URL = 'https: // github.com/silasabs/PIVIC-Comunicacoes-Opticas'
LICENSE = 'MIT'

# Função que carrega os requisitos necessários de um arquivo txt
def requirements(file):
    lineiter = (line.strip() for line in open(file))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(

    name = DISTNAME,
    maintainer = MAINTAINER,
    description = DESCRIPTION,
    license = LICENSE,
    url = URL,
    install_requires = requirements("./requirements.txt"),
    long_description = LONG_DESCRIPTION,
    classifiers = [
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    python_requires = '>=3.2',
)