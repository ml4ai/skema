from pathlib import Path
from setuptools import setup, find_namespace_packages
import os


version = "0.1"

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = "Python package for the SKEMA project"

setup(name='skema',
      version=version,
      description='Scientific Knowledge Extraction and Model Analysis',
      long_description=long_description,
      long_description_content_type='text',
      url='',
      author='Enrique Noriega, Adarsh Pyarelal, Clayton Morrison, Tito Ferra, Vincent Raymond',
      author_email='enoriega@arizona.edu',
      license='MIT',
      packages=find_namespace_packages(include=["skema.*"]),
      # package_dir={'mention_linking':os.path.join('skema', 'text_reading', 'mention_linking')},
      install_requires=[
        "gensim",
        "fastapi",
        "uvicorn",
        "dill",
        "networkx",
        "requests",
        "pygraphviz",
        "pytest"
      ],
      tests_require=["unittest"],
      zip_safe=False
)
