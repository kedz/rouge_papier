from setuptools import setup

setup(
   name='rouge_papier',
   version='0.0.1',
   description='A python ROUGE wrapper.',
   author='Chris Kedzie',
   author_email='kedzie@cs.columbia.edu',
   packages=["rouge_papier"],
   dependency_links = [],
   install_requires = ["pandas"],
)
