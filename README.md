# Kleinberg's Grid Ultimate


[![PyPI Status](https://img.shields.io/pypi/v/kleinberg-grid-simulator.svg)](https://pypi.python.org/pypi/kleinberg-grid-simulator)
[![Build Status](https://github.com/balouf/kleinberg-grid-simulator/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/balouf/kleinberg-grid-simulator/actions?query=workflow%3Abuild)
[![Documentation Status](https://github.com/balouf/kleinberg-grid-simulator/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/balouf/kleinberg-grid-simulator/actions?query=workflow%3Adocs)
[![License](https://img.shields.io/github/license/balouf/kleinberg-grid-simulator)](https://github.com/balouf/kleinberg-grid-simulator/blob/main/LICENSE)
[![Code Coverage](https://codecov.io/gh/balouf/kleinberg-grid-simulator/branch/main/graphs/badge.svg)](https://codecov.io/gh/balouf/kleinberg-grid-simulator/tree/main)

Great Trilogies Come in Threes.

- [Kleinberg's Grid Reloaded](https://inria.hal.science/hal-01417096/file/OPODIS2016-camera-ready-paper90.pdf) proposed
  a new *dynamic rejection sampling* approach to simulate
  [Kleinberg's small world model](http://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch20.pdf)
  on very large graphs.
- [Kleinberg's Grid Unchained](https://inria.hal.science/hal-02052607/document) introduced *double rejection sampling*,
  enabling computations on virtual grids the size of the universe (yes, that's big).
- [Kleinberg's Grid Ultimate](https://balouf.github.io/kleinberg-grid-simulator/) proposes an improved version of the
  simulator that isn't afraid with **universe to the square**! And it is nicely packaged for everyone to use.

- Free software: MIT license
- Documentation: https://balouf.github.io/kleinberg-grid-simulator/.


## Features

- Provides a Python frontend with possibility to use Julia or Python backend.
- Julia backend improved with fixed-size big ints (`int256`, `int512`, `int1024`) to speed up computation.
- Provides tools to parallelize, estimate complexity bounds, estimate *reasonable* shortcut distributions.
- Provides notebooks to benchmark the performance and reproduce the results from previous papers.


## Installation

Pip installation preferred.

```bash
pip install kleinberg-grid-simulator
```


## Usage

```python
from kleinberg_grid_simulator import compute_edt
compute_edt(n=1000, r=2, p=1, q=1)
```

## Credits

Céline Comte, co-author of [Kleinberg's Grid Unchained](https://inria.hal.science/hal-02052607/document).

This package was created with [Cookiecutter][CC] and the [Package Helper 3][PH3] project template.

[CC]: https://github.com/audreyr/cookiecutter
[PH3]: https://balouf.github.io/package-helper-3/
