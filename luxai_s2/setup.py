import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="luxai-s2",
    author="Lux AI Challenge",
    description="The Lux AI Challenge Season 2",
    license="MIT",
    keywords="reinforcement-learning machine-learning ai",
    url="https://github.com/Lux-AI-Challenge/Lux-Design-S2",
    long_description="Code for the Lux AI Challenge Season 2",
    packages=find_packages(exclude="kits"),
    entry_points={"console_scripts": ["luxai-s2 = luxai_runner.cli:main"]},
    version="2.1.7",
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pygame",
        "termcolor",
        "matplotlib",
        "pettingzoo",
        "vec_noise",
        "gym==0.21.0",
        "scipy",
        "importlib-metadata<5.0" # fixes bug where they deprecated an endpoint that openai gym uses
    ],
)
