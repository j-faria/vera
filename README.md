
<div align="center">
<img src="img/vera.gif" width="400px"></img>


[![PyPI version](https://badge.fury.io/py/verapy.svg)](https://pypi.org/project/verapy/)
[![CI](https://github.com/j-faria/vera/actions/workflows/python-package.yml/badge.svg)](https://github.com/j-faria/vera/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-informational.svg)](https://github.com/j-faria/vera/blob/main/LICENSE)
[![Funding](https://img.shields.io/badge/funding-FCT-darkgreen.svg)](https://www.fct.pt/)

</div>


### Installation

```
pip install verapy
```

### Getting started

```python
from vera import RV

s = RV('data_file.rdb')

s.plot()
```

### Interfacing with DACE

```python
from vera import DACE

HD10180 = DACE.HD10180
HD10180.plot_and_gls()
```

<img src="img/hd10180.png" width="80%"></img>

