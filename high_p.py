import numpy as np
import analytic_shrinkage as ana


if __name__ == '__main__':

    ps = np.arange(start=610, stop=620, step=50)

    print(ps)
    e_prial = np.zeros(len(ps))
    for i, p in enumerate(ps):
        reps = int(np.maximum(20, np.minimum(20, 1e5 / p)))
        prial = np.zeros(reps, dtype=float)
        for j in np.arange(reps):

            n = 600
            lam = np.concatenate([np.ones(int(p / 5)),
                                  3 * np.ones(int(2 * p / 5.)),
                                  10 * np.ones(int(2 * p / 5.))])
            sigma = np.diag(np.ones(p) * lam)
            xx = np.random.randn(n, p)
            _, u = np.linalg.eigh(sigma)
            y = np.linalg.solve(u.T, xx.T)
            s_tilde = ana.analytic_shrinkage(xx)
            s_sample = ana.sample_cov(xx)
            pr = ana.prial(s_sample, s_tilde, sigma)

            prial[j] = float(pr)

        e_prial[i] = np.mean(prial)
        print(np.mean(prial))

    print(e_prial)