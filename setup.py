from setuptools import setup, find_packages

setup(
    name='psyonic_playing_xylophone',
    version='0.1.0',
    packages=find_packages(
        where='.',
        exclude=['code', 'code.*', ],
    ),    
    python_requires='>=3.8',
    install_requires=[
        "wavio",
        "wandb",
        "numpy",
        "gymnasium",
        "pyaudio",
        "matplotlib",
        "librosa",
        "hydra-core",
        "tqdm"
    ],
)