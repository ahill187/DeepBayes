from setuptools import setup

setup(
          name='deep_bayes',
          version='1.0',
          description='Unfolding the W momentum',
          author='Ainsleigh Hill',
          author_email='ainsleigh.hill@alumni.ubc.ca',
          packages=['deep_bayes'],  #same as name
          install_requires=['ast', 'numpy', 'matplotlib', 'Keras'], #external packages as dependencies
          # scripts=[
          #          'scripts/cool',
          #          'scripts/skype',
          #         ]
)