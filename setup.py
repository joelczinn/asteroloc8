import setuptools

# Load version
__version__ = None
exec(open('asteroloc8/version.py').read())

# Load requirements
with open('requirements.txt') as file:
    requirements = file.read().splitlines()

# Setup
setuptools.setup(
    name='asteroloc8',
    version=__version__,
    description='Locate asteroseismic numax',
    packages=setuptools.find_packages(include=['asteroloc8', 'asteroloc8.*']),
    author='Alex Lyttle, Joel Zinn, Ted Mackereth, Jamie Tayar, Martin Nielsen',
    url='https://github.com/alexlyttle/asteroloc8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    include_package_data=True,  # <-- includes any package data without __init__.py
    python_requires='>=3.5',
    license='MIT',
)
