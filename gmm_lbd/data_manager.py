# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import OrderedDict
from itertools import cycle
import itertools

import numpy as np
import matplotlib as mpl

from scipy import linalg
# from scipy.stats import multivariate_normal

from sklearn import mixture
# from sklearn.utils.extmath import pinvh

from gmm import LbdGMM

import json


def plot_2D_mean_covars(means_covars, X=None, ax=None, color='b', size=0.3):
    """Regression datas (mean an covariance) of a GMM

    Parameters
    ----------
    X : data array of shape (n_samples, )

    means_covars : tuple of means and covariance
                    means data array of shape (n_samples, n_dimensions)
                    covars data array of shape (n_samples, n_dimensions, n_dimensions)

    ax : axis
        Matplotlib axis.

    color : string color af graph

    size : float between 0 and 1 related to alpha of fill_between and scalar of scatter
    """
    _, means, covars = means_covars
    X = X if X is not None else _

    scalar = 5 * size
    alpha = 0.8 * size

    if ax is None:
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)
        ax._get_lines.color_cycle = cycle(['r', 'g', 'b', 'c', 'm'])


    try:
        X.shape[1]
    except IndexError:
        X = X[:, np.newaxis]
    # means, covars = self.regression(X)
    ymax = np.empty(means.shape[0])
    ymin = np.empty(means.shape[0])

    ax.scatter(X[:, 0], means, scalar, color=color)
    for n in range(means.shape[0]):
        ymax[n] = means[n] + np.sqrt(covars[n][0])
        ymin[n] = means[n] - np.sqrt(covars[n][0])

    ax.fill_between(X[:, 0], ymin, ymax, alpha=alpha, color=color)

    return ax


class GmmManager(object):

    """ Provide an easy way to use GMMs, load and plot datasets"""

    def __init__(self, cv_types=['full'], n_components_range=range(1, 30), criteria='aic'):
        self.datasets = OrderedDict()
        self.gmms = OrderedDict()
        self.bics = OrderedDict()
        self.criteria = criteria
        # For BIC optimisation
        # ['full']#['spherical', 'tied', 'diag', 'spherical']
        self.cv_types = cv_types
        self.n_components_range = n_components_range

    def _check_gmm(self, dataset_name):
        if self.gmms[dataset_name] is None:
            self.gen_gmm(dataset_name)

    def add_dataset(self, data, name=None):
        """ Add a new dataset

            Parameters
            ----------
            data :array of shape(n_samples, n_dimensions)
            name : name of the dataset. Automaticly set to the len of the present dataset if None

            Returns
            -------
            return the name of the dataset in the dictionnary of datasets
        """
        name = str(len(self.datasets)) if name is None else name
        if name in self.datasets.keys() and not np.array_equal(self.datasets[name], data):
            self.datasets[name] = np.concatenate((self.datasets[name], data))
        else:
            self.datasets[name] = data
        self.gmms[name] = None
        return name

    def add_move(self, filename, suffix=''):
        """ Import a pypot MoveRecord file as a dict of array of shape(n_samples, n_dimensions)

            Parameters
            ----------
            filename : filename if a record file (specific to pypot)
            suffix : suffix added to the motor name

        """
        with open(filename, 'r') as move:
            d = json.load(move)
            timed_positions = d['positions']
            X = {}
            for i, timestamp in enumerate(sorted(timed_positions.keys())):
                dic = timed_positions[timestamp]
                for motor, values in dic.items():
                    try:
                        X[motor] = np.vstack(
                            (X[motor], np.array([[i, float(values[0])]])))
                    except KeyError:
                        X[motor] = np.array([[i, float(values[0])]])

        for k, v in X.items():
            self.add_dataset(v, name=k + suffix)

    def gen_gmm(self, dataset_name, cv_types=None, n_components_range=None, criteria=None):
        """ Generate a gmm to the current dataset optimizing de bic criteria

        Parameters
        ----------
        dataset_name : string
            dataset_name of the dataset to generate GMM

        cv_types : list
            types of covariances : 'diag', 'spherical', 'tied', or 'full'

        Returns
        -------
        return the generated GMM
        """
        cv_types = self.cv_types if cv_types is None else cv_types
        n_components_range = self.n_components_range if n_components_range is None else n_components_range
        criteria = self.criteria if criteria is None else criteria
        X = self.datasets[dataset_name]
        self.bics[dataset_name] = []
        lowest_bic = np.infty
        for cv_type in self.cv_types:
            for n_components in n_components_range:
                gmm = LbdGMM(n_components=n_components,
                             covariance_type=cv_type)
                gmm.fit(X)
                self.bics[dataset_name].append(
                    gmm.aic(X) if criteria == 'aic' else gmm.bic(X))
                if self.bics[dataset_name][-1] < lowest_bic:
                    lowest_bic = self.bics[dataset_name][-1]
                    best_gmm = gmm
        self.gmms[dataset_name] = best_gmm
        return best_gmm

    def plot_ellipses_and_samples(self, dataset_name=None, ax=None, colors=['r', 'g', 'b'], ):
        # Auto set data the dataset_name to the first inserted dataset
        dataset_name = self.datasets.keys()[0] if dataset_name is None else dataset_name
        self._check_gmm(dataset_name)
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)
        plt.title('GMM of {}'.format(dataset_name))
        ax.set_ylabel('angles of {}'.format(dataset_name))

        return self.gmms[dataset_name].plot_ellipses(self.datasets[dataset_name], ax=ax, colors=colors)

    def plot_regression(self, dataset_name=None, ax=None):
        dataset_name = self.datasets.keys()[0] if dataset_name is None else dataset_name
        self._check_gmm(dataset_name)
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)
        plt.title('Retrived datas of {}'.format(dataset_name))
        ax.set_ylabel('angles of {}'.format(dataset_name))

        X = np.linspace(min(self.datasets[dataset_name][:, 0]),
                        max(self.datasets[dataset_name][:, 0]),
                        100)
        gmm = self.gmms[dataset_name]
        return plot_2D_mean_covars(gmm.regression(X), ax=ax)


    def plot_bics(self, dataset_name=None, ax=None):
        """ Plot bic (bayesian information criterion) or aic for each 
            covariance_type/number of GMM compenents of the specified dataset
        """

        # Auto set data the dataset_name to the first inserted dataset
        dataset_name = self.datasets.keys()[0] if dataset_name is None else dataset_name

        # generate gmm if not already done
        self._check_gmm(dataset_name)

        bic = np.array(self.bics[dataset_name])
        color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
        bars = []

        # Plot the BIC scores
        if ax is None:
            plt.figure(figsize=(15, 5))
            spl = plt.subplot(2, 1, 1)
        else:
            spl = ax.subplot(2, 1, 1)

        for i, (cv_type, color) in enumerate(zip(self.cv_types, color_iter)):
            xpos = np.array(self.n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(self.n_components_range):
                                          (i + 1) * len(self.n_components_range)],
                                width=.2, color=color))
        plt.xticks(self.n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('{} score per model for {}'.format(
            self.criteria.upper(), dataset_name))
        xpos = np.mod(bic.argmin(), len(self.n_components_range)) + .65 +\
            .2 * np.floor(bic.argmin() / len(self.n_components_range))
        plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], self.cv_types)
        return ax


###############################################################################
# Old, not used anymore
###############################################################################

class SanitizeRecordsForGmm(object):

    """ Merge records data for gmm   """

    def __init__(self, dimension=0, cv_types=['full'], n_components_range=range(1, 30),):
        self.times = []
        self.dimension = dimension
        self.positions = []
        self.motors = []
        self.speeds = []
        self._X = []
        self.gmm = None
        self.__gmm_generated = False
        self.bics = []
        # For BIC optimisation
        # ['full']#['spherical', 'tied', 'diag', 'full']
        self.cv_types = cv_types
        self.n_components_range = n_components_range

    def add_record(self, times, positions, speeds=None):
        # For simple tests only
        if speeds is not None:
            raise NotImplementedError
        if len(times) != len(positions):
            raise Exception("times and positions must have same dimensions")
        if self.dimension != 0:
            if len(positions[0]) != self.dimension:
                raise Exception("positions dimension exceded")
        # Dirty Sort
        X1 = [i for i in zip(times, positions)]
        X2 = [i for i in zip(self.times, self.positions)]
        X = sorted(X1 + X2)
        self.positions = []
        self.times = []
        for i in X:
            self.times.append(i[0])
            self.positions.append(i[1])
    # def _gen_X(self):
    #    self._XX = np.array([self.times,self.positions]).transpose()

    def add_move(self, move_file):

        d = json.load(move_file)
        times = []
        positions = []
        speeds = []
        timed_positions = d['positions']
        for timestamp, mot_positions in timed_positions.items():
            motors_position = []
            motors_speed = []
            for motor, values in mot_positions.items():
                if motor not in self.motors:
                    self.motors.append(motor)
                motors_position.append(float(values[0]))
                motors_speed.append(float(values[1]))
            times.append(float(timestamp))
            positions.append(motors_position)
            speeds.append(motors_speed)
        # TODO: add speed
        self.add_record(times, positions)
        self.__gmm_generated = False

    def to_array(self):
        t_array = np.array([self.times]).transpose()
        if self.dimension == 0:
            pos_array = np.array([self.positions]).transpose()
        else:
            pos_array = np.array(self.positions)
        self._X = np.c_[t_array, pos_array]
        return self._X

    def plot(self, plot_type='raw', ax=None):
        # To be rewriten
        X = self.to_array()
        if plot_type == 'raw':
            if ax is None:
                plt.scatter(X[:, 0], X[:, 1], .8)
            else:
                ax.scatter(X[:, 0], X[:, 1], .8)
        elif plot_type == 'mean':
            sdt_dataset = []
            mean_dataset = []
            for i in range(len(X[:, 0])):
                sdt_dataset.append(np.std(X[i, 1]))
                mean_dataset.append(np.mean(X[i, 1]))

            Ytempmin = np.array(mean_dataset) - np.array(sdt_dataset)
            Ytempmax = np.array(mean_dataset) + np.array(sdt_dataset)
            plt.fill_between(X[:, 0], Ytempmin, Ytempmax, alpha=0.3)
            plt.plot(X[:, 0], np.array(mean_dataset))

    def gen_gmm(self, gmm_type='sklearn'):
        self.__gmm_generated = True
        X = self.to_array()
        lowest_bic = np.infty
        for cv_type in self.cv_types:
            for n_components in self.n_components_range:
                # Fit a mixture of Gaussians with EM
                if gmm_type == 'sklearn':
                    gmm = mixture.GMM(n_components=n_components,
                                      covariance_type=cv_type)
                else:
                    gmm = LbdGMM(n_components=n_components,
                                 covariance_type=cv_type)
                gmm.fit(X)
                self.bics.append(gmm.bic(X))
                if self.bics[-1] < lowest_bic:
                    lowest_bic = self.bics[-1]
                    best_gmm = gmm
        self.gmm = best_gmm
        return self.gmm

    def plot_bics(self, ax=None):
        if not self.__gmm_generated:
            self.gen_gmm()
        bic = np.array(self.bics)
        color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
        bars = []

        # Plot the BIC scores
        if ax is None:
            spl = plt.subplot(2, 1, 1)
        else:
            spl = ax.subplot(2, 1, 1)

        for i, (cv_type, color) in enumerate(zip(self.cv_types, color_iter)):
            xpos = np.array(self.n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(self.n_components_range):
                                          (i + 1) * len(self.n_components_range)],
                                width=.2, color=color))
        plt.xticks(self.n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC score per model')
        xpos = np.mod(bic.argmin(), len(self.n_components_range)) + .65 +\
            .2 * np.floor(bic.argmin() / len(self.n_components_range))
        plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], self.cv_types)
        return ax

    def plot_ellipses(self, ax=None, colors=None):
        if not self.__gmm_generated:
            self.gen_gmm()
        Y_ = self.gmm.predict(self._X)
        if colors is not None:
            colors = cycle(colors)
        else:
            colors = cycle(['r', 'g', 'b', 'c', 'm'])
        if ax is None:
            plt.figure(figsize=(15, 5))
            ax = plt.gca()

        for factor in np.linspace(0.5, 4.0, 8):
            for i, (mean, covar, color) in enumerate(zip(self.gmm.means_, self.gmm._get_covars(), colors)):

                if not np.any(Y_ == i):
                    continue
                plt.scatter(self._X[Y_ == i, 0], self._X[
                            Y_ == i, 1], .8, color=color)
                v, w = linalg.eigh(covar)
                u = w[0] / linalg.norm(w[0])
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                width, height = factor * np.sqrt(v)

                ell = Ellipse(xy=mean, width=width, height=height,
                              angle=angle)
                ell.set_alpha(0.25)
                if colors is not None:
                    ell.set_color(next(colors))
                ax.add_artist(ell)

    def pdfs(self):
        pass

    def plot_gmm(self, ax=None):
        if not self.__gmm_generated:
            self.gen_gmm()
        # To be improved
        color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
        # self.gen_gmm()
        Y_ = self.gmm.predict(self._X)
        if ax is None:
            ax = plt.subplot(1, 1, 1)

        for i, (mean, covar, color) in enumerate(zip(
                self.gmm.means_, self.gmm._get_covars(), color_iter)):
                # valeur propre, vecteur propre
            v, w = linalg.eigh(covar)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            ax.scatter(self._X[Y_ == i, 0], self._X[
                       Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(
                mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
