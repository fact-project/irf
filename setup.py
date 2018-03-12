from setuptools import setup


setup(
    name='irf',
    version='0.1.0',
    description='Functions to do instrument response functions for FACT',
    url='http://github.com/fact-project/irf',
    author='Kai Brügge, Maximilian Nöthe',
    author_email='kai.bruegge@tu-dortmund.de',
    license='MIT',
    packages=[
        'irf',
        'irf.scripts',
    ],
    install_requires=[
        'numpy',
        'eventio',
        'pandas',
        'click',
        'joblib',
        'pyfact>=0.9.1',
        'uncertainties',
    ],
    entry_points={
        'console_scripts': [
            'collect_corsika_run_headers=irf.scripts.collect_runheaders:main'
        ]
    },
    zip_safe=False,
)
