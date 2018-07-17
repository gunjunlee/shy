from setuptools import setup, find_packages

setup(
  name = 'shy',
  version = '0.1.0',
  description = 'very shy library',
  author = 'kevin970401',
  author_email = 'kevin970401@gmail.com',
  url = 'http://github.com/kevin970401/shy',
  install_requires = [],
  packages = find_packages(exclude = ['docs', 'tests*']),
  keywords = ['shy'],
  python_requires = '>=3',
  zip_safe = False,
  license = 'MIT'
)