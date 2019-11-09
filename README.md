# analytic_shrinkage
Provides a function that calculates an estimate of the covariance matrix shrunk using a non-linear analytic formula provided by
the working paper Ledoit and Wolf (2018), entitled ['Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices']
(http://www.econ.uzh.ch/static/wp/econwp264.pdf).


# Installation
```
pip install analytic_shrinkage
```

# Usage
```
import analytic_shrinkage as as
p = 2
n = 10
sigma = np.eye(p, p)
data = np.random.multivariate_normal(np.zeros(p), sigma, n)
sigma_tilde = as.analytic_shrinkage(data)
```
