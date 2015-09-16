# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import cycle

import numpy as np

from scipy import linalg
from scipy.stats import multivariate_normal

from sklearn import mixture
from sklearn.utils.extmath import pinvh


def invert_indices(n_features, indices):
    inv = np.ones(n_features, dtype=np.bool)
    inv[indices] = False
    inv, = np.where(inv)
    return inv


class LbdGMM(mixture.GMM):

    """ Override the default sklearn GMM mixture to add some regresssion features"""

    def __init__(self, n_components=1, covariance_type='full',
                 random_state=None, thresh=None, tol=1e-3, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc',
                 verbose=0):
        super(LbdGMM, self).__init__(n_components=n_components, covariance_type=covariance_type,
                                     random_state=random_state, thresh=thresh, tol=tol, min_covar=min_covar,
                                     n_iter=n_iter, n_init=n_init, params=params, init_params=init_params)

    # override fit to add the input dataset as a attribute (for conveniance)
    def fit(self, X):
        self.X_ = X
        return super(LbdGMM, self).fit(X)

    # override bic
    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        bic: float (the lower the better)
        """
        return (-2 * self.score(X).sum() +
                self._n_parameters() * 3 * np.log(X.shape[0]))

    def copy(self):
        """Create a copy of a LbdGMM object"""
        import copy
        return copy.deepcopy(self)


    def to_ellipses(self, factor=1.0):
        """Compute error ellipses.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        ellipses : array, shape (n_components, 4)
            Parameters that describe the error ellipses of all components:
            mean, widths, heights and angle.
        """

        ell_datas = []
        for (mean, covar) in zip(self.means_, self._get_covars()):
            v, w = linalg.eigh(covar)
            u = w[0] / linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            width, height = factor * np.sqrt(v)
            ell_datas.append((mean, width, height, angle))
        return ell_datas

    def conditional_distribution(self, x, indices=np.array([0])):
        """ Conditional gaussian distribution
            See
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions

            Return
            ------
            conditional : GMM
                Conditional GMM distribution p(Y | X=x)
        """
        n_features = self.means_.shape[1] - len(indices)
        expected_means = np.empty((self.n_components, n_features))
        expected_covars = np.empty((self.n_components, n_features, n_features))
        expected_weights = np.empty(self.n_components)

        # Highly inspired from https://github.com/AlexanderFabisch/gmr
        # Compute expexted_means, expexted_covars, given input X
        for i, (mean, covar, weight) in enumerate(zip(self.means_, self.covars_, self.weights_)):

            i1, i2 = invert_indices(mean.shape[0], indices), indices
            cov_12 = covar[np.ix_(i1, i2)]
            cov_11 = covar[np.ix_(i1, i1)]
            cov_22 = covar[np.ix_(i2, i2)]
            prec_22 = pinvh(cov_22)
            regression_coeffs = cov_12.dot(prec_22)

            if x.ndim == 1:
                x = x[:, np.newaxis]

            expected_means[i] = mean[i1] + regression_coeffs.dot((x - mean[i2]).T).T
            expected_covars[i] = cov_11 - regression_coeffs.dot(cov_12.T)
            expected_weights[i] = weight * \
                multivariate_normal.pdf(x, mean=mean[indices], cov=covar[np.ix_(indices, indices)])

        expected_weights /= expected_weights.sum()

        return expected_means, expected_covars, expected_weights

    def regression(self, X=None, indices=np.array([0])):
        """Predict approximed means and covariances

        Parameters
        ----------
        X : array, shape (n_samples, n_features_1)
            Values of the features that we know.

        indices : array, shape (n_features_1,)
            Indices of dimensions that we want to condition.

        Returns
        -------
        regression dataset

        approximed_means : array, shape (n_samples, n_features_2)
            Predicted means of missing values.

        approximed_covars : array, shape (n_samples, n_features_2)
            Predicted covariances of missing values.

        """

        # Automaticaly set the regression to the same range that the input datas
        X = X if X is not None else np.linspace(min(self.X_[:, 0]), max(self.X_[:, 0]), 100)

        try:
            n_samples, n_features_1 = X.shape
        except ValueError:
            X = X[:, np.newaxis]
            n_samples, n_features_1 = X.shape

        n_features_2 = self.means_.shape[1] - n_features_1
        approximed_means = np.empty((n_samples, n_features_2))
        approximed_covars = np.empty((n_samples, n_features_2, n_features_2))

        for i in range(n_samples):
            expected_means, expected_covars, expected_weights = self.conditional_distribution(X[i], indices)
            approximed_means[i] = expected_weights.dot(expected_means)
            approximed_covars[i] = pow(expected_weights, 2).dot(expected_covars.T[0].T)
        return X[:, 0], approximed_means, approximed_covars

    def pdf(self, X=None):
        """Compute probability density.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        p : array, shape (n_samples,)
            Probability densities of data.
        """
        X = X if X is not None else self.X_
        p = [multivariate_normal.pdf(X, mean=mean, cov=covar) for (mean, covar) in zip(self.means_, self._get_covars())]
        return np.dot(self.weights_, p)

    def plot_ellipses(self, X=None, ax=None, colors=['r', 'g', 'b', 'c', 'm'], ellipses_shapes=np.linspace(0.3, 2.0, 8)):
        """Plot error ellipses of GMM.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            MUST be 2 dimensions or None (TODO: generalize to many dimensions)

        ax : axis
            Matplotlib axis.

        colors : list
            list of colors

        ellipses_shapes : vector
            vector of ellipses factors shapes
        """
        X = X if X is not None else self.X_
        Y = self.predict(X)

        if colors is not None:
            colors = cycle(colors)

        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)

        for factor in ellipses_shapes:
            for i, (mean, width, height, angle) in enumerate(self.to_ellipses(factor)):
                if X is not None:
                    if not np.any(Y == i):
                        continue
                color = next(colors) if colors is not None else 'r'
                if X is not None:
                    plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)
                ell = Ellipse(xy=mean, width=width, height=height,
                              angle=np.degrees(angle))
                ell.set_alpha(0.25)
                ell.set_color(color)
                ax.add_artist(ell)
        return ax
