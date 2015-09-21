# coding: utf-8

# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# from collections import OrderedDict
# from itertools import cycle
# import itertools

import numpy as np
# import matplotlib as mpl

# from scipy import linalg
# from scipy.stats import multivariate_normal

# from sklearn import mixture
from sklearn.utils.extmath import pinvh
from gmm import LbdGMM


def prod(X=None, *gmms):
    """Predict a generalized trajectory from multiple independant constraints.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Range of regression values
    gmms :

    """

    # Ugly way to make X optional
    if 'LbdGMM' in str(type(X)):
        gmms += (X,)
        X = np.empty(0)
        for gmm in gmms:
            X = X if gmm.X_.shape[0] < X.shape[0] else np.linspace(min(gmm.X_[:, 0]), max(gmm.X_[:, 0]), 200)

    all_means = []
    all_covars = []
    for gmm in gmms:
        _, m, c = gmm.regression(X)
        all_means.append(m)
        all_covars.append(c)

    generalized_mean = np.empty_like(all_means[0])
    generalized_covar = np.empty_like(all_covars[0])

    for k in range(X.shape[0]):
        generalized_covar[k] = pinvh(sum([pinvh(cov[k]) for cov in all_covars]))
        generalized_mean[k] = generalized_covar[k].dot(
            sum([pinvh(cov[k]).dot(means[k]) for cov, means in zip(all_covars, all_means)]))

    return X, generalized_mean, generalized_covar


def influence_gmm(gmm, coef=1.0):
    gmm = gmm.copy()
    gmm.covars_ /= coef
    return gmm


def seq(gmm1, gmm2, wait=0, ver='add'):
    # TODO : play with gmm.weights_
    # TODO : generalise with an unknown number of gmms
    gmm1 = gmm1.copy()
    gmm2 = gmm2.copy()
    if ver == 'add':
        delta = max(gmm1.X_[:, 0]) - min(gmm2.X_[:, 0])
        delta += wait
        gmm2.X_[:, 0] += delta
        gmm2.means_[:, 0] += delta
    elif ver == 'align':
        delta = min(gmm1.X_[:, 0]) - min(gmm2.X_[:, 0])
        delta += wait
        gmm2.X_[:, 0] += delta
        gmm2.means_[:, 0] += delta

    gmm_output = LbdGMM(n_components=gmm1.n_components + gmm2.n_components)
    gmm_output._set_covars(np.concatenate((gmm1._get_covars(), gmm2._get_covars())))
    gmm_output.means_ = np.concatenate((gmm1.means_, gmm2.means_))
    gmm_output.weights_ = np.concatenate(
        (gmm1.weights_ * gmm1.n_components / (gmm1.n_components + gmm2.n_components),
         gmm2.weights_ * gmm2.n_components / (gmm1.n_components + gmm2.n_components)))
    gmm_output.X_ = np.concatenate((gmm1.X_, gmm2.X_))
    return gmm_output

# def gmm_product_old(X, gmm1, gmm2):
#     """Predict a generalized trajectory from two independant constraints.

#     Parameters
#     ----------
#     X : array-like, shape (n_samples, n_features)
#         Data.

#     """
#     means1, covars1 = gmm1.regression(X)
#     means2, covars2 = gmm2.regression(X)

#     generalized_mean = np.empty_like(means1)
#     generalized_covar = np.empty_like(covars1)
#     for i, (mean1, covar1, mean2, covar2) in enumerate(zip(means1, covars1, means2, covars2)):
#         generalized_mean[i] = pinvh(pinvh(covar1) + pinvh(covar2)).dot(
#             pinvh(covar1).dot(mean1) + pinvh(covar2).dot(mean2))
#         generalized_covar[i] = pinvh(pinvh(covar1) + pinvh(covar2))

#     return generalized_mean, generalized_covar
