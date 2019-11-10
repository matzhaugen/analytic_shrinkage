# Non-Linear Shrinkage
Provides a function that calculates an estimate of the covariance matrix shrunk using a non-linear analytic formula provided by
the working paper Ledoit and Wolf (2018), entitled ['Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices']
(http://www.econ.uzh.ch/static/wp/econwp264.pdf).


# Installation
```
pip install nonlinshrink
```

# Usage
```
import numpy as np
import nonlinshrink as nls
p = 2
n = 12
sigma = np.eye(p, p)
data = np.random.multivariate_normal(np.zeros(p), sigma, n)
sigma_tilde = nls.shrink_cov(data)
```
# Developing
Please submit a PR! The shrinkage function itself is located in `nonlinshrink.py`. 
For running the tests do 
```
git clone https://github.com/matzhaugen/analytic_shrinkage.git
cd analytic_shrinkage
pip install -e . # install the package
pytest
```
