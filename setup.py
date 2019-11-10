# Create Whl: python setup.py sdist bdist_wheel
# Local installation: python -m pip install dist/[name-of-whl]
# Push to pip: python -m twine upload dist/*

import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='nonlinshrink',
    version='0.5',
    author="Matz Haugen",
    author_email="matzhaugen@gmail.com",
    description="Non-Linear Shrinkage Estimator from Ledoit and Wolf (2018) ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matzhaugen/analytic_shrinkage",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
)
