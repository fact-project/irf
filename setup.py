from setuptools import setup, find_packages


setup(
    name='irf',
    version='0.0.4',
    description='Functions to do instrument response functions for FACT',
    url='http://github.com/fact-project/irf',
    author='Kai Brügge, Maximilian Nöthe',
    author_email='kai.bruegge@tu-dortmund.de',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'eventio',
        'pandas',
        'click',
        'joblib',
        'pyfact>=0.20.1',
        'uncertainties',
        'corsikaio',
    ],
    entry_points={
        'console_scripts': [
            'fact_read_corsika_headers=irf.scripts.read_corsika_headers:main'
        ]
    },
    zip_safe=False,
)
