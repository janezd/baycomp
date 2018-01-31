import os
import pickle

import numpy as np
from scipy import stats


__all__ = ["two_on_multiple", "SignTest", "SignedRankTest", "HierarchicalTest",
           "two_on_single", "CorrelatedT"]


# Common utility functions
# ------------------------

try:
    import seaborn
    seaborn.set(color_codes=True)
except ImportError:
    pass


def import_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("Plotting requires 'matplotlib'; "
                          "use 'pip install {1}' to install it")


def _check_args(x, y, rope=0, prior=1, nsamples=50000):
    if x.ndim != 1:
        raise ValueError("'x' must be a 1-dimensional array")
    if y.ndim != 1:
        raise ValueError("'y' must be a 1-dimensional array")
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must be of same length")
    if rope < 0:
        raise ValueError('Rope width cannot be negative')
    if prior < 0:
        raise ValueError('Prior strength cannot be negative')
    if not round(nsamples) == nsamples > 0:
        raise ValueError('Number of samples must be a positive integer')


# Two classifiers on multiple data sets
# -------------------------------------

class Sample:
    def __init__(self, sample, names=None):
        self.sample = sample
        self.names = names

    def p_values(self, with_rope=True):
        winners = np.argmax(self.sample, axis=1)
        pl, pe, pr = np.bincount(winners, minlength=3) / len(winners)
        if with_rope:
            return pl, pe, pr
        return pl, pr

    def plot(self, names=None):
        if np.max(self.sample[:, 1]) < 0.1:
            return self.plot_histogram(names)
        else:
            return self.plot_simplex(names)

    def plot_simplex(self, names=None):
        plt = import_plt()
        from matplotlib.lines import Line2D

        def project(points):
            from math import sqrt, sin, cos, pi
            p1, p2, p3 = points.T / sqrt(3)
            x = (p2 - p1) * cos(pi / 6) + 0.5
            y = p3 - (p1 + p2) * sin(pi / 6) + 1 / (2 * sqrt(3))
            return np.vstack((x, y)).T

        names = names or self.names or ("C1", "C2")
        pl, pe, pr = self.p_values()

        vert0 = project(np.array(
            [[0.3333, 0.3333, 0.3333],
             [0.5, 0.5, 0],
             [0.5, 0, 0.5],
             [0, 0.5, 0.5]]))

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')

        # triangle
        ax.add_line(
            Line2D([0, 0.5, 1.0, 0],
                   [0, np.sqrt(3) / 2, 0, 0], color='orange'))
        # decision lines
        for i in (1, 2, 3):
            ax.add_line(
                Line2D([vert0[0, 0], vert0[i, 0]],
                       [vert0[0, 1], vert0[i, 1]], color='orange'))

        ax.text(0, -0.04,
                'p({}) = {:.3f}'.format(names[0], pl),
                horizontalalignment='center', verticalalignment='top')
        ax.text(0.5, np.sqrt(3) / 2,
                'p(rope) = {:.3f}'.format(pe),
                horizontalalignment='center', verticalalignment='bottom')
        ax.text(1, -0.04,
                'p({}) = {:.3f}'.format(names[1], pr),
                horizontalalignment='center', verticalalignment='top')

        # project and draw points
        tripts = project(self.sample[:, [0, 2, 1]])
        # pylint: disable=no-member
        plt.hexbin(tripts[:, 0], tripts[:, 1], mincnt=1, cmap=plt.cm.Blues_r)
        # Leave some padding around the triangle for vertex labels
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.axis('off')
        return fig

    def plot_histogram(self, names):
        plt = import_plt()

        names = names or self.names or ("C1", "C2")
        points = self.sample[:, 2]
        pr = np.sum(points > 0.5) + 0.5 * np.sum(points == 0.5)
        pl = 1 - pr
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.hist(points, 50, color="#34ccff")
        ax.axis(xmin=0, xmax=1)
        ax.text(0, 0, "\np({}) = {:.3f}".format(names[0], pl),
                horizontalalignment='left', verticalalignment='top')
        ax.text(1, 0, "\np({}) = {:.3f}".format(names[1], pr),
                horizontalalignment='right', verticalalignment='top')
        ax.get_yaxis().set_ticklabels([])
        ax.axvline(x=0.5, color="#ffad2f", linewidth=2)
        return fig


class Test:
    LEFT, ROPE, RIGHT = range(3)

    def __new__(cls, x, y, rope, *args, nsamples=50000, **kwargs) -> Sample:
        return Sample(cls.sample(x, y, rope, nsamples=nsamples, *args, **kwargs))

    @classmethod
    def sample(cls, x, y, rope, *args, nsamples=50000, **kwargs):
        pass


    @classmethod
    def test(cls, x, y, rope, *args, **kwargs):
        # pylint: disable=no-member
        return cls(x, y, rope, *args, **kwargs).p_values(rope > 0)

    @classmethod
    def plot(cls, x, y, rope, *args, names=None, **kwargs):
        # pylint: disable=no-value-for-parameter
        return cls(x, y, rope, *args, **kwargs).plot(names)

    @classmethod
    def plot_simplex(cls, x, y, rope, *args, names=None, **kwargs):
        # pylint: disable=no-value-for-parameter
        return cls(x, y, rope, *args, **kwargs).plot_simplex(names)

    @classmethod
    def plot_histogram(cls, x, y, rope, *args, names=None, **kwargs):
        # pylint: disable=no-value-for-parameter
        return cls(x, y, rope, *args, **kwargs).plot_simplex(names)


class SignTest(Test):
    @classmethod
    # pylint: disable=arguments-differ
    def sample(cls, x, y, rope=0, prior=1, nsamples=50000):
        if isinstance(prior, tuple):
            prior, prior_place = prior
        else:
            prior_place = cls.ROPE
        _check_args(x, y, rope, prior, nsamples)

        diff = y - x
        nleft = sum(diff < -rope)
        nright = sum(diff > rope)
        nrope = len(diff) - nleft - nright
        alpha = np.array([nleft, nrope, nright], dtype=float)
        alpha += 0.0001  # for numerical stability
        alpha[prior_place] += prior
        return np.random.dirichlet(alpha, nsamples)  # pylint: disable=no-member


class SignedRankTest(Test):
    @classmethod
    # pylint: disable=arguments-differ
    def sample(cls, x, y, rope=0, nsamples=50000):
        def heaviside(a, thresh):
            return (a > thresh).astype(float) + (a == thresh).astype(float) * 0.5

        def diff_sums(x, y):
            diff = np.hstack(([0], y - x))
            diff_m = np.lib.stride_tricks.as_strided(
                diff, strides=diff.strides + (0,), shape=diff.shape * 2)
            weights = np.ones(len(diff))
            weights[0] = 0.5
            return diff_m + diff_m.T, weights

        def with_rope():
            sums, weights = diff_sums(x, y)
            above_rope = heaviside(sums, 2 * rope)
            below_rope = heaviside(-sums, 2 * rope)
            samples = np.zeros((nsamples, 3))
            # pylint: disable=no-member
            for i, samp_weights in enumerate(np.random.dirichlet(weights, nsamples)):
                prod_weights = np.outer(samp_weights, samp_weights)
                samples[i, 0] = np.sum(prod_weights * below_rope)
                samples[i, 2] = np.sum(prod_weights * above_rope)
            samples[:, 1] = -samples[:, 0] - samples[:, 2] + 1
            return samples

        def without_rope():
            sums, weights = diff_sums(x, y)
            above_0 = heaviside(sums, 0)
            samples = np.zeros((nsamples, 3))
            # pylint: disable=no-member
            for i, samp_weights in enumerate(np.random.dirichlet(weights, nsamples)):
                prod_weights = np.outer(samp_weights, samp_weights)
                samples[i, 2] = np.sum(prod_weights * above_0)
            samples[:, 0] = -samples[:, 2] + 1
            return samples

        _check_args(x, y, rope, nsamples)
        return with_rope() if rope > 0 else without_rope()


class HierarchicalTest(Test):
    @classmethod
    # pylint: disable=arguments-differ
    # pylint: disable=unused-argument
    def sample(cls, x, y, rope, rho, std_upper_bound=1000, chains=4, nsamples=None):
        CACHE_FILE_NAME = "stored-stan-results.pickle"

        def try_unpickle(fname):
            try:
                with open(fname, "rb") as f:
                    return pickle.load(f)
            except:  # pylint: disable=bare-except
                pass

        def try_pickle(obj, fname):
            try:
                with open(fname, "wb") as f:
                    pickle.dump(obj, f)
            except PermissionError:
                pass

        def arr_to_tuple(a):
            return tuple(tuple(e for e in row) for row in a)

        def arg_hash():
            return hash((arr_to_tuple(x), arr_to_tuple(y), rho, std_upper_bound, chains))

        def get_cached_results():
            nonlocal delta0, std0, nu
            hash_results = try_unpickle(CACHE_FILE_NAME)
            if hash_results is not None and hash_results[0] == arg_hash():
                delta0, std0, nu = hash_results[1:]  # pylint: disable=unused-variable
                return True
            return False

        def get_stan_model(model):
            try:
                import pystan
            except ImportError:
                raise ImportError("Hierarchical model uses 'pystan'; "
                                  "install it by 'pip install pystan'")

            stan_file = os.path.join(os.path.split(__file__)[0], model)
            pickle_file = stan_file + ".pickle"
            if os.path.exists(pickle_file) and \
                    os.path.getmtime(pickle_file) > os.path.getmtime(stan_file):
                model = try_unpickle(pickle_file)
                if model is not None:
                    return model
            model_code = open(stan_file).read()
            stan_model = pystan.StanModel(model_code=model_code)
            try_pickle(stan_model, pickle_file)
            return stan_model

        def run_stan():
            nonlocal delta0, std0, nu, rope

            diff = y - x
            stds = np.std(diff, axis=1)
            std_diff = np.mean(stds)
            diff /= std_diff
            rope /= std_diff

            nscores_2 = nscores // 2
            for sample, std in zip(diff, stds):
                if std == 0:
                    noise = np.random.uniform(-rope, rope, nscores_2)  # pylint: disable=no-member
                    sample[:nscores_2] = noise
                    sample[nscores_2:] = -noise

            std_within = np.mean(np.std(diff, axis=1))  # may be different from std_diff!
            std_among = np.std(np.mean(diff, axis=1)) if ndatasets > 1 else std_within

            stan_data = dict(
                x=diff, Nsamples=nscores, q=ndatasets,
                rho=rho,
                deltaLow=-1 / std_diff, deltaHi=1 / std_diff,
                stdLow=0, stdHi=std_within * std_upper_bound,
                std0Low=0, std0Hi=std_among * std_upper_bound,
            )

            model = get_stan_model("hierarchical-t-test.stan")
            fit = model.sampling(data=stan_data, chains=chains)
            results = fit.extract(permuted=True)
            delta0 = results["delta0"]
            std0 = results["std0"]
            nu = results["nu"]
            try_pickle((arg_hash(), delta0, std0, nu), CACHE_FILE_NAME)


        delta0 = std0 = nu = None
        ndatasets, nscores = x.shape
        if not get_cached_results():
            run_stan()
        samples = np.empty((len(nu), 3))
        for mu, std, df, sample_row in zip(delta0, std0, nu, samples):
            sample_row[2] = 1 - stats.t.cdf(rope, df, mu, std)
            sample_row[0] = stats.t.cdf(-rope, df, mu, std)
            sample_row[1] = 1 - sample_row[0] - sample_row[2]
        return samples


# Two classifiers on a single data set
# ------------------------------------

class TDistribution:
    def __init__(self, meanx, meany, mean, var, df, rope, *, names=None):
        self.meanx = meanx
        self.meany = meany
        self.mean = mean
        self.var = var
        self.df = df
        self.rope = rope
        self.names = names

    def p_values(self):
        if self.var == 0:
            return (float(self.mean < self.rope),
                    float(abs(self.mean) <= self.rope),
                    float(self.rope < self.mean))
        pr = 1 - stats.t.cdf(self.rope, self.df, self.mean, np.sqrt(self.var))
        pl = stats.t.cdf(-self.rope, self.df, self.mean, np.sqrt(self.var))
        if self.rope == 0:
            return pl, pr
        return pl, 1 - pl - pr, pr

    def plot(self, names=None):
        plt = import_plt()
        names = names or self.names or ("C1", "C2")

        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_xlabel("difference ({}: {:.3f}, {}: {:.3f})".format(
            names[0], self.meanx, names[1], self.meany))
        ax.get_yaxis().set_ticklabels([])
        ax.axvline(x=-self.rope, color="#ffad2f", linewidth=2, label="rope")
        ax.axvline(x=self.rope, color="#ffad2f", linewidth=2)

        targs = (self.df, self.mean, np.sqrt(self.var))
        xs = np.linspace(min(stats.t.ppf(0.005, *targs), -1.05 * self.rope),
                         max(stats.t.ppf(0.995, *targs), 1.05 * self.rope),
                         100)
        ys = stats.t.pdf(xs, *targs)
        ax.plot(xs, ys, color="#2f56e0", linewidth=2, label="pdf")
        ax.fill_between(xs, ys, np.zeros(100), color="#34ccff")
        ax.legend()
        return fig


class CorrelatedT:
    """
    Correlated t-test

    The class uses the Bayesian interpretation of the p-value and returns
    the probabilities the difference are below `-rope`, within `[-rope, rope]`
    and above the `rope`. For details, see `A Bayesian approach for comparing
    cross-validated algorithms on multiple data sets
    <http://link.springer.com/article/10.1007%2Fs10994-015-5486-z>`_,
    G. Corani and A. Benavoli, Mach Learning 2015.

    The test assumes that the classifiers were evaluated using cross
    validation. The number of folds is determined from the length of the vector
    of differences, as `len(diff) / runs`. The variance includes a correction
    for underestimation of variance due to overlapping training sets, as
    described in `Inference for the Generalization Error
    <http://link.springer.com/article/10.1023%2FA%3A1024068626366>`_,
    C. Nadeau and Y. Bengio, Mach Learning 2003.).

    Args:
        x (np.array): a vector of scores for the first model
        y (np.array): a vector of scores for the second model
        rope (float): the width of the rope (default: 0)
        runs (int): number of repetitions of cross validation (default: 1)
    """
    def __new__(cls, x, y, rope=0, runs=1):
        _check_args(x, y, rope)
        if not int(runs) == runs > 0:
            raise ValueError('Number of runs must be a positive integer')
        if len(x) % round(runs) != 0:
            raise ValueError("Number of measurements is not divisible by number of runs")

        diff = y - x
        n = len(diff)
        nfolds = n / runs
        mean = np.mean(diff)
        var = np.var(diff, ddof=1)
        var *= 1 / n + 1 / (nfolds - 1)  # Nadeau-Bengio's correction
        # pylint: disable=no-member
        return TDistribution(np.mean(x), np.mean(y), mean, var, n - 1, rope)

    @classmethod
    def test(cls, x, y, rope=0, runs=1):
        return cls(x, y, rope, runs).p_values()  # pylint: disable=no-member

    @classmethod
    def plot(cls, x, y, rope=0, runs=1, *, names=None):
        return cls(x, y, rope, runs).plot(names)  # pylint: disable=no-value-for-parameter


# Shortcuts
# ---------

def _test_results(test, x, y, rope, names, *args, plot=False, **kwargs):
    sample = test(x, y, rope, *args, **kwargs)
    if plot:
        return sample.p_values(), sample.plot(names)
    else:
        return sample.p_values()


def two_on_single(x, y, rope, runs=1, names=None, *, plot=False):
    return _test_results(CorrelatedT, x, y, rope, names, plot, runs=runs)


def two_on_multiple(x, y, rope, *args,
                    names=None, hierarchical=False, plot=False, **kwargs):
    test = HierarchicalTest if hierarchical else SignedRankTest
    return _test_results(test, x, y, rope, names=names, plot=plot, *args, **kwargs)
