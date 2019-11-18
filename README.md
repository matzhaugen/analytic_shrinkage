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
n = 13
sigma = np.eye(p, p)
data = np.random.multivariate_normal(np.zeros(p), sigma, n)
sigma_tilde = nls.shrink_cov(data)
```
The data is automatically demeaned unless otherwise specified. In the case where the data has been pre-processed and the effective degrees of freedom of the dataset is decreased, e.g. through an OLS regression, the user can specify this through a parameter `k` which signifies the degrees of freedom already subtracted. For example,
```
import numpy as np
import nonlinshrink as nls
p = 2
n = 14
sigma = np.eye(p, p)
data = np.random.multivariate_normal(np.zeros(p), sigma, n) + np.arange(n)[:, np.newaxis] + 1
x = np.vstack((np.ones(n).T, np.arange(n).T)).T
betahat = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, data))
datahat = np.dot(x, betahat)
res = data - datahat
sigma_tilde = nls.shrink_cov(res, k=2)  # corresponding to 2 degrees of freedom
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
