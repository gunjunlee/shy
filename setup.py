from setuptools import setup, find_packages

setup(
    name='shy',
    version='0.2.4.dev2021102600',
    description='very shy library',
    author='gunjunlee',
    author_email='gunjunlee97@gmail.com',
    url='http://github.com/gunjunlee/shy',
    install_requires=['backtrace', 'tqdm'],
    packages=find_packages(exclude=['docs', 'tests*']),
    keywords=['shy'],
    python_requires='>=3',
    zip_safe=False,
    license='MIT'
)
