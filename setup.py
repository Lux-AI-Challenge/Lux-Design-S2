import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="luxai2022",
    version="0.1.0",
    author="Lux AI Challenge Nonprofit",
    description="The Lux AI Challenge Season 2",
    license="MIT",
    keywords="reinforcement-learning machine-learning ai",
    url="https://github.com/Lux-AI-Challenge/Lux-Design-2022",
    packages=["luxai2022", "tests", "luxai_runner"],
    long_description=read("README.md"),
)
