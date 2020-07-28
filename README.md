# LyricPsych

General descriptions goes here


## Install

current alpha version is only for internal use, not for public usage.

To install this package, we recommend you use the python virtual environment. Inside the virtualenv, installation is using `pip` and `git`.

```console
$ pip install git+https://github.com/mmc-tudelft/lyricpsych.git@packaging
```
feature extractor `lyricpsych-extract` installed along with the package. The usage of the `lyricpsych-extractor` can be found in the `-h` option. For instance, you can extract `personality`, `value`, `topic`, `linguistic` features by using the example below:

```console
$ lyricpsych-extract \
    /path/to/the/lyrics_data.csv \
    /path/for/output/ \
    --w2v glove-twitter-25 \
    --features linguistic value liwc topic
```



## TODO list

- [x] refactoring
  - [x] split extractor to dedicated extractors
  - [x] minor refactorings
  - [x] clean up
    - [x] unused functions
    - [x] unused data files
    - [x] unused scripts
  - [x] restructuring
    - [x] split `task` to the separate sub-module
    - [x] separate `fm` and `als_feat` to the separate repo
- [ ] Documentation
  - [ ] docstrings
  - [ ] landing page description
  - [ ] doc generation
- [x] features
  - [x] experimental run reproduction cli
- [ ] deploy
  - [ ] writing testings
  - [ ] CI [Travis integration]
  - [ ] register to PyPI

## Reference

TBD
