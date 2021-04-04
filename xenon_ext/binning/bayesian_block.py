#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-04
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np


def get_bayesian_blocks_binning_boundary(x: np.ndarray, y=None) -> np.ndarray:
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD from
    https://jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    x : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    x = np.sort(x)
    N = x.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([x[:1],
                            0.5 * (x[1:] + x[:-1]),
                            x[-1:]])
    block_length = x[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    for K in range(N):
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4
        fit_vec[1:] += best[:K]

        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return np.array(sorted(list(set(edges[change_points]))))


if __name__ == '__main__':
    # Define our test distribution: a mix of Cauchy-distributed variables
    from scipy import stats

    np.random.seed(0)
    x = np.concatenate([stats.cauchy(-5, 1.8).rvs(500),
                        stats.cauchy(-4, 0.8).rvs(2000),
                        stats.cauchy(-1, 0.3).rvs(500),
                        stats.cauchy(2, 0.8).rvs(1000),
                        stats.cauchy(4, 1.5).rvs(500)])

    # truncate values to a reasonable range
    x = x[(x > -15) & (x < 15)]
    import pylab as plt

    # plt.hist(x, bins=100)
    # plot a standard histogram in the background, with alpha transparency
    H1 = plt.hist(x, bins=200, histtype='stepfilled',
                  alpha=0.2, normed=True)
    # plot an adaptive-width histogram on top
    H2 = plt.hist(x, bins=get_bayesian_blocks_binning_boundary(x), color='black',
                  histtype='step', normed=True)
    plt.show()
