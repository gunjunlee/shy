git tag -a v0.2.4 -m "Release version 0.2.4"
python setup.py bdist_wheel
twine upload dist/shy-[*EDIT*].whl