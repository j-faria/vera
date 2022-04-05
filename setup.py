""" vera """

from setuptools import setup

setup(
    name='verapy',
    version='0.0.5',
    description='velocidades radiais',
    url='https://github.com/j-faria/vera',
    author='João Faria',
    author_email='joao.faria@astro.up.pt',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    packages=['vera'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'requests',
        'tqdm',
        'python-dace-client',
        'cached-property',
        'colorful',
        'astropy',
        'astroquery',
        'lightkurve',
        'gatspy',
        'PyAstronomy',
        'pyexoplaneteu',
        'pyephem',
        'mpldatacursor',
        'iCCF',
    ],
    dependency_links=[
        'https://dace.unige.ch/api/python-dace-client'
    ],
    package_data={'vera': ['cute_style.mplstyle']},
    zip_safe=False,
)
