import matplotlib.pyplot as plt
import numpy as np
import scipy
import itertools

from scipy import linalg
import matplotlib as mpl

from sklearn import mixture
import json


class Sanitize_records_for_gmm(object):

    """ Merge records data for gmm   """

    def __init__(self, dimension=0, cv_types=['spherical', 'tied', 'diag', 'full'], n_components_range=range(1, 30)):
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
                plt.scatter(X[:, 0], X[:, 1])
            else:
                ax.scatter(X[:, 0], X[:, 1])
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

    def gen_gmm(self):
        self.__gmm_generated = True
        X = self.to_array()
        lowest_bic = np.infty
        for cv_type in self.cv_types:
            for n_components in self.n_components_range:
                # Fit a mixture of Gaussians with EM
                gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
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
