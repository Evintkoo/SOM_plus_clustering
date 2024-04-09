from setuptools import setup, find_packages

setup(
    name='som_plus_clustering',
    version='0.1.0',
    description='A description of my module',
    author='Evint Leovonzko',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.2',
        'pandas==2.1.4',
        'python-dateutil==2.8.2',
        'pytz==2023.3.post1',
        'six==1.16.0',
        'tzdata==2023.3',
    ],
)