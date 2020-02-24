# Build status

[![Build
Status](https://travis-ci.org/brianjaywoo/bmi_203_w2020_hw3.svg?branch=master)](https://travis-ci.org/brianjaywoo/bmi_203_w2020_hw3)

```
```

##  Using skeleton

To use the package, first make a new conda environment and activate it

```
conda create -n exampleenv python=3
source activate exampleenv
```

then run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `example/__main__.py`) can be run as follows, and
accepts two sequences along with a scoring matrix. Run the following command
to see the structure of how to call the main module:

```

python -m smith_waterman

```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.
