"""This code replicates the baseline results for the analytic shrinkage estimator
from the working paper called
'Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices'
Ledoit and Wolf 2018
Figure  4
"""

import numpy as np
import nonlinshrink as nls


if __name__ == '__main__':

    ps = np.arange(start=10, stop=200, step=20)

    print(ps)
    e_prial = np.zeros(len(ps))
    for i, p in enumerate(ps):
        reps = int(np.maximum(100, np.minimum(100, 1e5 / p)))
        prial = np.zeros(reps, dtype=float)
        for j in np.arange(reps):

            n = 600
            lam = np.concatenate([np.ones(int(p / 5)),
                                  3 * np.ones(int(2 * p / 5.)),
                                  10 * np.ones(int(2 * p / 5.))])
            sigma = np.diag(np.ones(p) * lam)
            xx = np.random.randn(n, p)
            d, u = np.linalg.eigh(sigma)
            # y = np.linalg.solve(u.T, xx.T)
            s_sqrt = np.eye(p, p) * np.sqrt(lam)
            y = np.dot(xx, s_sqrt)
            s_tilde = nls.shrink_cov(y)
            s_sample = np.cov(y.T)
            pr = ana.prial(s_sample, s_tilde, sigma)

            prial[j] = float(pr)

        e_prial[i] = np.mean(prial)
        print(np.mean(prial))

    print(e_prial)
