import numpy as np
from numpy import matlib as ml


def analytic_shrinkage(xx):
    # % extract sample eigenvalues sorted in ascending order and eigenvectors
    n, p = xx.shape

    # % important:
    # sample size n must be >= 12
    assert n >= 12
    sample = np.dot(xx.T, xx) / n
    lam, u = np.linalg.eigh(sample)
    # compute analytical nonlinear shrinkage kernel formula
    lam = lam[np.maximum(0, p - n): p]
    L = ml.repmat(lam.T, np.minimum(p, n), 1).T
    h = np.power(n, -1 / 3.)
    # % Equation(4.9)
    H = h * L.T
    x = (L - L.T) / H
    ftilde = (3 / 4. / np.sqrt(5)) * np.mean(np.maximum(1 - x ** 2. / 5., 0) / H, 1)
    # % Equation(4.7)
    Hftemp = (-3 / 10 / np.pi) * x + (3 / 4. / np.sqrt(5) / np.pi) * (1 - x ** 2. / 5.) \
        * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    # % Equation(4.8)
    Hftemp[np.abs(x) == np.sqrt(5)] = (-3 / 10 / np.pi) * x[np.abs(x) == np.sqrt(5)]
    Hftilde = np.mean(Hftemp / H, 1)
    if p <= n:
        dtilde = lam / ((np.pi * (p / n) * lam * ftilde) ** 2
                        + (1 - (p / n) - np.pi * (p / n) * lam * Hftilde) ** 2)
    # % Equation(4.3)
    else:
        Hftilde0 = (1 / np.pi) * (3 / 10. / h ** 2 + 3 / 4. / np.sqrt(5) / h * (1 - 1 / 5. / h ** 2)
                                  * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))) * np.mean(1 / lam)
        # % Equation(C.8)
        dtilde0 = 1 / (np.pi * (p - n) / n * Hftilde0)
        # % Equation(C.5)
        dtilde1 = lam / (np.pi ** 2 * lam ** 2. * (ftilde ** 2 + Hftilde ** 2))
        # % Eq. (C.4)
        dtilde = np.array([dtilde0 * np.ones(p - n, 1), dtilde1])

    sigmatilde = np.dot(np.dot(u, np.diag(dtilde)), u.T)

    return sigmatilde
