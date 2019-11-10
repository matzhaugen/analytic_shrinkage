"""This code replicates the high dimensional results for the analytic
shrinkage estimator from the working paper called
'Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices'
Ledoit and Wolf 2018
Figure  4
"""

import numpy as np
import nonlinshrink as nls

CONST = 120000


def high_p():

    c = np.arange(start=1.1, stop=10, step=1)

    print(c)
    e_prial = np.zeros(len(c))
    for i, ci in enumerate(c):
        n = int(np.sqrt(CONST / ci))
        p = int(CONST / n / 2) * 2
        reps = int(np.maximum(100, np.minimum(100, 1e5 / p)))
        print(reps)
        prial = np.zeros(reps, dtype=float)
        for j in np.arange(reps):
            first = int(p / 5)
            second = int(2 * p / 5.)
            third = p - first - second
            lam = np.concatenate([np.ones(first),
                                  3 * np.ones(second),
                                  10 * np.ones(third)])
            sigma = np.diag(np.ones(p) * lam)
            xx = np.random.randn(n, p)
            d, u = np.linalg.eigh(sigma)
            # y = np.linalg.solve(u.T, xx.T)
            s_sqrt = np.eye(p, p) * np.sqrt(lam)
            y = np.dot(xx, s_sqrt)
            s_tilde = nls.analytic_shrinkage(y)
            s_sample = np.cov(y.T)
            pr = nls.prial(s_sample, s_tilde, sigma)

            prial[j] = float(pr)

        e_prial[i] = np.mean(prial)
        print(np.mean(prial))

    print(e_prial)
