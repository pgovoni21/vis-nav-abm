from distutils.core import setup

from setuptools import find_packages

setup(
    name='abm',
    description='ABM to simulate navigation relying on spatial + social visual cues',
    url='https://github.com/pgovoni21/vis-nav-abm',
    maintainer='Patrick Govoni @ HU ITB, Collective Information Processing Lab',
    packages=find_packages(exclude=['tests']),
    package_data={'abm': ['*.txt']},
    python_requires=">=3.7",
    install_requires=[
        'pygame',
        'python-dotenv',
        'numpy',
        'matplotlib',
        'opencv-python', # screenrecorder for sims
        'colorcet', # plot_funcs.py
        'scipy', # trajs.py
        'seaborn', # trajs.py

        # 'pip install torch --index-url https://download.pytorch.org/whl/cpu' # doesn't work with setup
        # 'pip install git+https://github.com/nnaisense/pgpelib.git#egg=pgpelib' # also

        # also need to run the Solution 2 commands here:
        # https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris
    ],
    entry_points={
        'console_scripts': [
            'abm=abm.start_sim:start',
            'EA=abm.start_EA:start_EA',
            'multi=abm.start_EA_multirun:EA_runner',
        ]
    },
)
