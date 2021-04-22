from setuptools import setup

setup(name='dfencoding',
      version='0.1',
      description='A simple library for encoding and decoding pandas DataFrame.',
      url='https://github.com/Lpourchot/dfencoding',
      author='Laurent Pourchot',
      author_email='laurent_pourchot@fr.ibm.com',
      license='GNU General Public License v3.0',
      packages=['dfencoding'],
      install_requires=[
          'numpy',
          'pandas'
      ],
      zip_safe=False)