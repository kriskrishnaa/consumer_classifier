from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='consumer_classifier',
      version='0.1',
      description='Classifies consumers based on behaviour pattern',
      url='http://github.com/kriskrishnaa',
      author='Krishna Babu',
      author_email='moorthyk66@gmail.com',
      license='MIT',
      packages=['ConsumerClassifier'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      zip_safe=False)
