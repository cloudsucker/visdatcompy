from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="visdatcompy",
    version="0.3.2",
    author="cls, Reape4er, Adg1r",
    author_email="ferjenkill@gmail.com",
    description="Библиотека для сравнения визуальных наборов данных",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxfraid/visdatcompy",
    packages=find_packages(),
    install_requires=requirements,
)