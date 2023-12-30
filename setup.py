from distutils.core import setup

from setuptools import find_packages

setup(
    name='abm',
    description='Agent based model to simulate forage agents relying on spatial + social visual cues',
    url='https://github.com/pgovoni21/DavidMezey-PyGame-ABM',
    maintainer='Patrick Govoni @ HU ITB, Collective Information Processing',
    packages=find_packages(exclude=['tests']),
    package_data={'abm': ['*.txt']},
    python_requires=">=3.7",
    install_requires=[
        'pygame',
        'python-dotenv',
        'numpy',
        'zarr',
        'matplotlib',
        # 'cma',
        'opencv-python', # screenrecorder

        # 'pip install torch --index-url https://download.pytorch.org/whl/cpu' # doesn't work with setup

        # 'pip install git+https://github.com/nnaisense/pgpelib.git#egg=pgpelib' # doesn't work with setup # now doesn't work at all --> local install below
        # cd ../pgpelib
        # pip install -e .
    ],
    # extras_require={
    #     'test': [
    #         'bandit',
    #         'flake8',
    #         'pytest',
    #         'pytest-cov'
    #     ]
    # },
    entry_points={
        'console_scripts': [
            'abm=abm.start_sim:start',
            'EA=abm.start_EA:start_EA',
            'multi=abm.start_EA_multirun:EA_runner',
        ]
    },
    # classifiers=[
    #     'Development Status :: 2 - Pre-Alpha',
    #     'Intended Audience :: Science/Research',
    #     'Topic :: Scientific/Engineering :: Artificial Intelligence',
    #     'Operating System :: Other OS',
    #     'Programming Language :: Python :: 3.8'
    # ],
    # test_suite='tests',
    # zip_safe=False
)
