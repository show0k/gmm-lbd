# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import OrderedDict

import numpy as np
import itertools
from itertools import cycle


from scipy import linalg
import matplotlib as mpl

from sklearn import mixture
import json


class LbdGMM(mixture.GMM):
    """ Override the default sklearn GMM mixture to add some other tools"""

    def __init__(self, n_components=1, covariance_type='diag',
                 random_state=None, thresh=None, tol=1e-3, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc',
                 verbose=0):
        super(LbdGMM, self).__init__(n_components=n_components, covariance_type=covariance_type,
                                     random_state=random_state, thresh=thresh, tol=tol, min_covar=min_covar,
                                     n_iter=n_iter, n_init=n_init, params=params, init_params=init_params)

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

    def plot_ellipses(self, X, ax=None, colors=['r', 'g', 'b','c','m'], ellipses_shapes=np.linspace(0.3, 2.0, 8)):
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
        from matplotlib.patches import Ellipse
        from itertools import cycle

        Y = self.predict(X) if X is not None else None

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


class GmmManager(object):
    """ Provide an easy way to use GMMs, load and plot datasets"""

    def __init__(self, cv_types=['full'], n_components_range=range(1, 30), criteria='aic'):
        self.datasets = OrderedDict()
        self.gmms = OrderedDict()
        self.bics = OrderedDict()
        self.criteria = criteria
        # For BIC optimisation
        self.cv_types = cv_types  # ['full']#['spherical', 'tied', 'diag', 'full']
        self.n_components_range = n_components_range

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

    def add_move(self,filename):
        """ Import a pypot MoveRecord file as a dict of array of shape(n_samples, n_dimensions)

            Parameters
            ----------
            filename : path if a record file (specific to pypot)

        """
        with open(filename, 'r') as move:
            d = json.load(move)
            timed_positions = d['positions']
            X = {}
            for i, timestamp in enumerate(sorted(timed_positions.keys())):
                dic = timed_positions[timestamp]
                for motor, values in dic.items():
                    try:
                        X[motor] = np.vstack((X[motor], np.array([[i, float(values[0])]])))
                    except KeyError:
                        X[motor] = np.array([[i, float(values[0])]])

            for k, v in X.items():
                self.add_dataset(v, name=k)

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
                gmm = LbdGMM(n_components=n_components, covariance_type=cv_type)
                gmm.fit(X)
                self.bics[dataset_name].append(gmm.aic(X) if criteria == 'aic' else gmm.bic(X))
                if self.bics[dataset_name][-1] < lowest_bic:
                    lowest_bic = self.bics[dataset_name][-1]
                    best_gmm = gmm
        self.gmms[dataset_name] = best_gmm
        return best_gmm

    def plot_ellipses_and_samples(self, dataset_name=None, ax=None, colors=['r', 'g', 'b']):
        # Auto set data the dataset_name to the first inserted dataset
        dataset_name = self.datasets.keys()[0] if dataset_name is None else dataset_name
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax = plt.gca()
        plt.title('GMM of {}'.format(dataset_name))
        ax.set_ylabel('positions of {}'.format(dataset_name))
        
        # generate gmm if not already done
        if self.gmms[dataset_name] is None:
            self.gen_gmm(dataset_name)
        return self.gmms[dataset_name].plot_ellipses(self.datasets[dataset_name], ax=ax, colors=colors)



    # TO BE REMOVED
    def plot_ellipses(self, means, covars,  xlim=(0, 10), ylim=(0, 10), ax=None, colors=['r', 'g', 'b'], elipses_shapes=np.linspace(0.8, 4.0, 5)):
        # Auto set data the dataset_name to the first insert item
        colors = cycle(colors)
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax = plt.gca()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        for factor in elipses_shapes:
            for i, (mean, covar, color) in enumerate(zip(means, covars, colors)):

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

        return ax

    def plot_bics(self, dataset_name=None, ax=None):
        """ Plot bic (bayesian information criterion) or aic for each 
            covariance_type/number of GMM compenents of the specified dataset
        """
        # Auto set data the dataset_name to the first inserted dataset
        dataset_name = self.datasets.keys()[0] if dataset_name is None else dataset_name

        # generate gmm if not already done
        if self.gmms[dataset_name] is None:
            self.gen_gmm(dataset_name)

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
        plt.title('{} score per model for {}'.format(self.criteria.upper(), dataset_name))
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
        self.cv_types = cv_types  # ['full']#['spherical', 'tied', 'diag', 'full']
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
                    gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
                else:
                    gmm = LbdGMM(n_components=n_components, covariance_type=cv_type)
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
                plt.scatter(self._X[Y_ == i, 0], self._X[Y_ == i, 1], .8, color=color)
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
            ax.scatter(self._X[Y_ == i, 0], self._X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
