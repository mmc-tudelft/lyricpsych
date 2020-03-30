from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f]


setup(name='lyricpsych',
      version='0.0.1',
      description='Psychological text features for music lyrics',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='Psychological text features for music lyrics',
      url='http://github.com/mmc-tudelft/lyricpsych',
      author='Jaehun Kim',
      author_email='j.h.kim@tudelft.nl',
      license='MIT',
      packages=['lyricpsych'],
      install_requires=requirements(),
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      entry_points={
          'console_scripts': ['lyricpsych-extract=lyricpsych.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)
