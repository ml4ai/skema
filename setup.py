from pathlib import Path
from setuptools import setup, find_packages
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
    #   packages=find_packages('src'),
      package_dir={'mention_linking':os.path.join('skema', 'text_reading', 'mention_linking')},
      install_requires=[
        "gensim",
        #"automates @ https://github.com/danbryce/automates/archive/e5fb635757aa57007615a75371f55dd4a24851e0.zip#sha1=f9b3c8a7d7fa28864952ccdd3293d02894614e3f"
      ],
      tests_require=["unittest"],
      zip_safe=False
)