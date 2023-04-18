from setuptools import setup, find_packages

"""
python3 -m unittest
vim setup.py
rm -rf dist/
python3 setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
"""


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="t5maru",
    version="0.2.2",
    license="MIT",
    author="Kimio Kuramitsu",
    description="Deep Learning",
    url="https://github.com/kuramitsu/t5maru",
    packages=["t5maru", "t5maru.metrics"],
    package_dir={"t5maru": ""},
    package_data={"t5maru": ["*/*"]},
    install_requires=_requires_from_file("requirements.txt"),
    entry_points={
        "console_scripts": [
            "t5maru=t5maru.t5tune:main",
            "t5train=t5maru.t5tune:main_train",
            "t5test=t5maru.t5tune:main_test",
            "t5score=t5maru.t5score:main",
            "t5cp=t5maru.t5utils:main_cp",
            "t5new=t5maru.t5utils:main_new",
            "t5len=t5maru.t5utils:main_len",
            #              "t5dump=t5maru.t5dump:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Education",
    ],
)
