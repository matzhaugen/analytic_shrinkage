import numpy as np
import nonlinshrink as nls
from numpy import prod
import pytest


def factorial(n):
    return prod(range(1, n + 1))


def n_choose_k(n, k):
    return factorial(n) / factorial(k) / factorial(n - k)


def test_analytic_shrinkage():
    """Runs the most simple possible test: 2 variables with 12 samples.
    """
    x = np.array([[-0.98511153, -0.2599713],
                  [0.20374114, -0.59699234],
                  [0.28570754, 0.77542166],
                  [-0.78768299, 0.45808448],
                  [0.80494623, -2.16308943],
                  [-2.01761751, -0.27944538],
                  [-0.24710646, -0.33260174],
                  [0.369508, 0.57123535],
                  [2.07913481, -0.0749686],
                  [1.07194672, -0.69343179],
                  [-0.11947645, -1.45457297],
                  [-0.54877366, -1.21191041]])

    expected = np.array([[1.04344299, 0.0335051],
                         [0.0335051, 1.11112703]])
    sigma_tilde = nls.shrink_cov(x, 0)

    np.testing.assert_allclose(sigma_tilde, expected)


def test_demean():
    """Runs the high-dimensional case.
    """
    p = 2
    n = 13
    sigma = np.eye(p, p)
    data = np.random.multivariate_normal(np.zeros(p), sigma, n)

    sigma_tilde = nls.shrink_cov(data)
    S = np.sum(sigma_tilde[np.eye(p) == 0]) / n_choose_k(p, 2) / np.sum(np.diag(sigma_tilde)) * p
    assert S < 1  # assert that the diagonal is the major contributor


def test_ols():
    """Runs the high-dimensional case.
    """
    p = 2
    n = 14
    sigma = np.eye(p, p)
    data = np.random.multivariate_normal(np.zeros(p), sigma, n) + np.arange(n)[:, np.newaxis] + 1
    x = np.vstack((np.ones(n).T, np.arange(n).T)).T
    betahat = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, data))
    datahat = np.dot(x, betahat)
    res = data - datahat
    sigma_tilde = nls.shrink_cov(res, k=2)  # corresponding to 2 degrees of freedom
    S = np.sum(sigma_tilde[np.eye(p) == 0]) / n_choose_k(p, 2) / np.sum(np.diag(sigma_tilde)) * p
    assert S < 1  # assert that the diagonal is the major contributor


def test_singular():
    """Runs the high-dimensional case.
    """
    p = 2
    n = 13
    sigma = np.eye(p, p)
    data = np.random.multivariate_normal(np.zeros(p), sigma, n)
    data = np.hstack((data, data))
    with pytest.raises(ValueError):
        sigma_tilde = nls.shrink_cov(data, 0)


def test_large_p():
    """Runs the high-dimensional case.
    """
    p = 13
    n = 12
    sigma = np.eye(p, p)
    data = np.random.multivariate_normal(np.zeros(p), sigma, n)

    sigma_tilde = nls.shrink_cov(data, 0)
    S = np.sum(sigma_tilde[np.eye(p) == 0]) / n_choose_k(p, 2) / np.sum(np.diag(sigma_tilde)) * p
    assert S < 1  # assert that the diagonal is the major contributor
