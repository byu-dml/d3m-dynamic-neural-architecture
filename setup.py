from setuptools import setup, find_packages

from dna import __version__

setup(name='dna',
      packages=find_packages(include=['dna', 'dna.*']),
      version=__version__,
      description='A collection of models used for pipeline selection',
      author='Brandon Schoenfeld, Orion Weller, Mason Poggemann, Even Peterson, Erik Huckvale',
      url='https://github.com/byu-dml/d3m-dynamic-neural-architecture/tree/develop/dna',
      keywords=['metalearning', 'metafeature', 'machine learning', 'metalearn', 'dna'],
      install_requires=[
            'pandas==0.24.2',
            'scikit-learn==0.21.1',
            'scipy==1.3.0',
            'numpy==1.16.3',
            'torch==1.1.0',
            'tqdm==4.32.1'
      ])

