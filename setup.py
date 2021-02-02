from setuptools import setup

setup(
          name='deep_bayes',
          version='1.0',
          description='Unfolding the W boson momentum',
          author='Ainsleigh Hill',
          author_email='ainsleigh.hill@alumni.ubc.ca',
          packages=['deep_bayes'],
          install_requires=['ast', 'numpy', 'matplotlib', 'Keras']
)
