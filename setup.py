from setuptools import setup

setup(

    name='socraticbumpsearch', 
    version='0.0.1', 
    description='Socratic bump search meta estimator, scikit-learn compatible',
    url='https://github.com/pr38/socraticbumpsearch', 
    install_requires=["scikit-learn>=0.20.1"],
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3',
    ],
    packages=["socraticbumpsearch"],
    python_requires='>=3.5',
)
