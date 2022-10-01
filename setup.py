from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'matplotlib',
    'rich',
]

setup(
    name='nmpc_cgmres',
    version='0.0.0',
    author='Dohi',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=install_requires,
)
