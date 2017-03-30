from setuptools import setup


setup(
    name='irf',
    version='0.0.1',
    description='Functions to do instrument response functions for FACT',
    url='http://github.com/fact-project/irf',
    author='Kai Brügge, Maximilian Nöthe',
    author_email='kai.bruegge@tu-dortmund.de',
    license='MIT',
    packages=[
        'irf',
    ],
    install_requires=[
        'numpy',
        'eventio',
        'pandas',
        'click',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'collect_corsika_run_headers=irf.scripts.collect_runheaders:main'
        ]
    },
    zip_safe=False,
)
