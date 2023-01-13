import os

from setuptools import find_packages, setup

with open(os.path.join("luxai_s2", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="luxai-s2",
    author="Lux AI Challenge",
    description="The Lux AI Challenge Season 2",
    license="MIT",
    keywords="reinforcement-learning machine-learning ai",
    url="https://github.com/Lux-AI-Challenge/Lux-Design-S2",
    packages=find_packages(exclude="kits"),
    package_data={"luxai-s2": ["version.txt"]},
    long_description=read("../README.md"),
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["luxai-s2 = luxai_runner.cli:main"]},
    version=__version__,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pygame",
        "termcolor",
        "matplotlib",
        "pettingzoo",
        "vec_noise",
        "omegaconf",
        "gym==0.19",
        "scipy",
    ],
)
