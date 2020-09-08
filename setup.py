from setuptools import setup

# Load version
__version__ = None
exec(open('asteroloc8/version.py').read())

# Load requirements
with open('requirements.txt') as file:
    requirements = file.read().splitlines()

# Setup
setup(
    name='asteroloc8',
    version=__version__,
    description='Locate asteroseismic oscillations',
    packages=['asteroloc8'],
    package_dir={
        'asteroloc8': 'asteroloc8',
        },
    author='Ted Mackereth, Alex Lyttle, Joel Zinn',
    install_requires=requirements,
    include_package_data=True,  # <-- includes any package data without __init__.py
)
