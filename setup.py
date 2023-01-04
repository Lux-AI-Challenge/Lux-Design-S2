import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="luxai2022",
    version="1.1.6",
    author="Lux AI Challenge",
    description="The Lux AI Challenge Season 2",
    license="MIT",
    keywords="reinforcement-learning machine-learning ai",
    url="https://github.com/Lux-AI-Challenge/Lux-Design-2022",
    packages=find_packages(exclude="kits"),
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    entry_points={'console_scripts': [
        'luxai2022 = luxai_runner.cli:main']},
    install_requires=['numpy', 'pygame', 'termcolor', 'matplotlib', 'pettingzoo', 'vec_noise', 'omegaconf', 'gym==0.19', 'scipy'],
)
