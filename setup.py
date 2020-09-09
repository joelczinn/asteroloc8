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
    description='Locate asteroseismic oscillations',
    packages=setuptools.find_packages(include=['asteroloc8', 'asteroloc8.*']),
    author='Ted Mackereth, Alex Lyttle, Joel Zinn, Mat Schofield, William Chaplin, Jamie Tayar',
    install_requires=requirements,
    include_package_data=True,  # <-- includes any package data without __init__.py
)
