from setuptools import setup, find_packages

'''
python3 -m unittest
vim setup.py
rm -rf dist/
python3 setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
'''


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(name="trainmaru",
      version="0.0.5",
      license='MIT',
      author='Kimio Kuramitsu',
      description="Deep Learning",
      url="https://github.com/kkuramitsu/kogi",
      packages=['trainmaru', 'trainmaru.metrics'],
      package_dir={"trainmaru": ""},
      package_data={'trainmaru': ['*/*']},
      install_requires=_requires_from_file('requirements.txt'),
      entry_points={
          "console_scripts": [
              "t5train=trainmaru.t5train:main",
              "t5test=trainmaru.t5train:main_test",
              "t5score=trainmaru.t5score:main",
          ]
      },
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Intended Audience :: Education',
      ],
      )
