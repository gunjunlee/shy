from setuptools import setup, find_packages

setup(
  name='shy',
  version='0.2.4.dev2019112200',
  description='very shy library',
  author='kevin970401',
  author_email='kevin970401@gmail.com',
  url='http://github.com/kevin970401/shy',
  install_requires=['backtrace', 'tqdm'],
  packages=find_packages(exclude=['docs', 'tests*']),
  keywords=['shy'],
  python_requires='>=3',
  zip_safe=False,
  license='MIT'
)