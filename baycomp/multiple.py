import os
import pickle

import numpy as np
from scipy import stats

from .utils import check_args, seaborn_plt, call_shortcut

__all__ = ["SignTest", "SignedRankTest", "HierarchicalTest", "two_on_multiple"]

class Posterior:
    """
    Sampled posterior distribution

    Args:
        sample (np.array): a 3 x `nsamples` array
        names (tuple of str or None): names of learning algorithms
            (default: `None`)
    """
    def __init__(self, sample, *, names=None):
        self.sample = sample
        self.names = names

    def probs(self, with_rope=True):
        """
        Compute and return probabilities

        Args:
            with_rope (bool): tells whether the sample includes the
                probabilities for the rope region (default: `True`)

        Returns:
            `(p_left, p_rope, p_right)` if `with_rope=True`;
            otherwise `(p_left, p_right)`.
        """
        winners = np.argmax(self.sample, axis=1)
        pl, pe, pr = np.bincount(winners, minlength=3) / len(winners)
        return (pl, pe, pr) if with_rope else (pl / (pl + pr), pr / (pl + pr))

    def plot(self, names=None):
        """
        Plot the posterior distribution.

        If there are samples in which the probability of `rope` is higher
        than 0.1, the distribution is shown in a simplex
        (see :obj:`plot_simplex`), otherwise as a histogram
        (:obj:`plot_histogram`).

        Args:
            names (tuple of str or `None`): names of classifiers

        Returns:
            matplotlib figure
        """
        if np.max(self.sample[:, 1]) < 0.1:
            return self.plot_histogram(names)
        else:
            return self.plot_simplex(names)

    def plot_simplex(self, names=None):
        """
        Plot the posterior distribution in a simplex.

        The distribution is shown as a triangle with regions corresponding to
        first classifier having higher scores than the other by more than rope,
        the second having higher scores, or the difference being within the
        rope.

        Args:
            names (tuple of str): names of classifiers

        Returns:
            matplotlib figure
        """

        with seaborn_plt() as plt:
            from matplotlib.lines import Line2D

            def project(points):
                from math import sqrt, sin, cos, pi
                p1, p2, p3 = points.T / sqrt(3)
                x = (p2 - p1) * cos(pi / 6) + 0.5
                y = p3 - (p1 + p2) * sin(pi / 6) + 1 / (2 * sqrt(3))
                return np.vstack((x, y)).T

            fig, ax = plt.subplots()
            ax.set_aspect('equal', 'box')

            # triangle
            ax.add_line(Line2D([0, 0.5, 1.0, 0],
                               [0, np.sqrt(3) / 2, 0, 0], color='orange'))
            names = names or self.names or ("C1", "C2")
            pl, pe, pr = self.probs()
            ax.text(0, -0.04,
                    'p({}) = {:.3f}'.format(names[0], pl),
                    horizontalalignment='center', verticalalignment='top')
            ax.text(0.5, np.sqrt(3) / 2,
                    'p(rope) = {:.3f}'.format(pe),
                    horizontalalignment='center', verticalalignment='bottom')
            ax.text(1, -0.04,
                    'p({}) = {:.3f}'.format(names[1], pr),
                    horizontalalignment='center', verticalalignment='top')
            cx, cy = project(np.array([[0.3333, 0.3333, 0.3333]]))[0]
            for x, y in project(np.array([[.5, .5, 0],
                                          [.5, 0, .5],
                                          [0, .5, .5]])):
                ax.add_line(Line2D([cx, x], [cy, y], color='orange'))

            # project and draw points
            tripts = project(self.sample[:, [0, 2, 1]])
            plt.hexbin(tripts[:, 0], tripts[:, 1],
                       mincnt=1, cmap=plt.cm.Blues_r)
            # Leave some padding around the triangle for vertex labels
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-0.2, 1.2)
            ax.axis('off')
            return fig

    def plot_histogram(self, names):
        """
        Plot the posterior distribution as histogram.

        Args:
            names (tuple of str): names of classifiers

        Returns:
            matplotlib figure
        """

        with seaborn_plt() as plt:
            names = names or self.names or ("C1", "C2")
            points = self.sample[:, 2]
            pr = (np.sum(points > 0.5) + 0.5 * np.sum(points == 0.5)) \
                / len(points)
            pl = 1 - pr
            fig, ax = plt.subplots()
            ax.grid(True)
            ax.hist(points, 50, color="#34ccff")
            ax.axis(xmin=0, xmax=1)
            ax.text(0, 0, "\n\np({}) = {:.3f}".format(names[0], pl),
                    horizontalalignment='left', verticalalignment='top')
            ax.text(1, 0, "\n\np({}) = {:.3f}".format(names[1], pr),
                    horizontalalignment='right', verticalalignment='top')
            ax.get_yaxis().set_ticklabels([])
            ax.axvline(x=0.5, color="#ffad2f", linewidth=2)
            return fig


class Test:
    LEFT, ROPE, RIGHT = range(3)

    def __new__(cls, x, y, rope=0, *, nsamples=50000, random_state=None, **kwargs):
        return Posterior(cls.sample(x, y, rope, nsamples=nsamples, random_state=random_state, **kwargs))

    @classmethod
    def sample(cls, x, y, rope=0, nsamples=50000, **kwargs):
        """
        Compute a sample of posterior distribution.

        Derived classes override this method to implement specific
        sampling methods. Derived methods may have additional arguments.

        Args:
            x (np.array): a vector of scores for the first model
            y (np.array): a vector of scores for the second model
            rope (float): the width of the region of practical equivalence (default: 0)
            nsamples (int): the number of samples (default: 50000)

        Returns:
            np.array of shape (`nsamples`, 3)
        """

    @classmethod
    def probs(cls, x, y, rope=0, *, nsamples=50000, **kwargs):
        """
        Compute and return probabilities

        Args:
            x (np.array): a vector of scores for the first model
            y (np.array): a vector of scores for the second model
            rope (float): the width of the region of practical equivalence (default: 0)
            nsamples (int): the number of samples (default: 50000)

        Returns:
            `(p_left, p_rope, p_right)` if `rope > 0`;
            otherwise `(p_left, p_right)`.
        """
        # new returns an instance of Posterior, not Test
        # pylint: disable=no-value-for-parameter
        return cls(x, y, rope, nsamples=nsamples, **kwargs).probs(rope > 0)

    @classmethod
    def plot(cls, x, y, rope, *, nsamples=50000, names=None, **kwargs):
        """
        Plot the posterior distribution.

        If there are samples in which the probability of `rope` is higher
        than 0.1, the distribution is shown in a simplex
        (see :obj:`plot_simplex`), otherwise as a histogram
        (:obj:`plot_histogram`).

        Args:
            x (np.array): a vector of scores for the first model
            y (np.array): a vector of scores for the second model
            rope (float): the width of the region of practical equivalence (default: 0)
            nsamples (int): the number of samples (default: 50000)
            names (tuple of str): names of classifiers

        Returns:
            matplotlib figure
        """
        # pylint: disable=no-value-for-parameter
        return cls(x, y, rope, nsamples=nsamples, **kwargs).plot(names)

    @classmethod
    def plot_simplex(cls, x, y, rope, *, nsamples=50000, names=None, **kwargs):
        """
        Plot the posterior distribution in a simplex.

        The distribution is shown as a triangle with regions corresponding to
        first classifier having higher scores than the other by more than rope,
        the second having higher scores, or the difference being within the
        rope.

        Args:
            x (np.array): a vector of scores for the first model
            y (np.array): a vector of scores for the second model
            nsamples (int): the number of samples (default: 50000)
            names (tuple of str): names of classifiers

        Returns:
            matplotlib figure
        """
        # pylint: disable=no-value-for-parameter
        return cls(x, y, rope, nsamples=nsamples, **kwargs).plot_simplex(names)

    @classmethod
    def plot_histogram(cls, x, y, *, nsamples=50000, names=None, **kwargs):
        """
        Plot the posterior distribution as histogram.

        Args:
            x (np.array): a vector of scores for the first model
            y (np.array): a vector of scores for the second model
            nsamples (int): the number of samples (default: 50000)
            names (tuple of str): names of classifiers

        Returns:
            matplotlib figure
        """
        # pylint: disable=no-value-for-parameter
        return cls(x, y, rope=0, nsamples=nsamples, **kwargs)\
            .plot_histogram(names)


class SignTest(Test):
    """
    Compute a Bayesian sign test
    (`A Bayesian Wilcoxon signed-rank test based on the Dirichlet process
    <http://proceedings.mlr.press/v32/benavoli14.pdf>`_,
    A. Benavoli et al, ICML 2014).

    Argument `prior` can give a strength (as `float`) of a prior put on the
    rope region, or a tuple with prior's position and strength, for instance
    `(SignTest.LEFT, 1.0)`. Position can be `SignTest.LEFT`, `SignTest.ROPE`
    or `SignTest.RIGHT`.
    """

    @classmethod
    # pylint: disable=arguments-differ
    def sample(cls, x, y, rope=0, *, prior=1, nsamples=50000, random_state=None):
        if isinstance(prior, tuple):
            prior, prior_place = prior
        else:
            prior_place = cls.ROPE
        check_args(x, y, rope, prior, nsamples)

        _random_state = np.random.RandomState(random_state)

        diff = y - x
        nleft = sum(diff < -rope)
        nright = sum(diff > rope)
        nrope = len(diff) - nleft - nright
        alpha = np.array([nleft, nrope, nright], dtype=float)
        alpha += 0.0001  # for numerical stability
        alpha[prior_place] += prior
        return _random_state.dirichlet(alpha, nsamples)


class SignedRankTest(Test):
    """
    Compute a Bayesian signed-rank test
    (`A Bayesian Wilcoxon signed-rank test based on the Dirichlet process
    <http://proceedings.mlr.press/v32/benavoli14.pdf>`_,
    A. Benavoli et al, ICML 2014).

    Arguments `x` and `y` should be one-dimensional arrays with average
    performances across data sets. These can be obtained using any sampling
    method, not necessarily cross validation.
    """

    @classmethod
    # pylint: disable=arguments-differ
    def sample(cls, x, y, rope=0, *, prior=0.5, nsamples=50000, random_state=None):
        def heaviside(a, thresh):
            return (a > thresh).astype(float) + 0.5 * (a == thresh).astype(float)

        def diff_sums(x, y):
            diff = np.hstack(([0], y - x))
            diff_m = np.lib.stride_tricks.as_strided(
                diff, strides=diff.strides + (0,), shape=diff.shape * 2)
            weights = np.ones(len(diff))
            weights[0] = prior
            return diff_m + diff_m.T, weights

        def with_rope():
            sums, weights = diff_sums(x, y)
            above_rope = heaviside(sums, 2 * rope)
            below_rope = heaviside(-sums, 2 * rope)
            samples = np.zeros((nsamples, 3))
            _random_state = np.random.RandomState(random_state)
            for i, samp_weights in enumerate(_random_state.dirichlet(weights, nsamples)):
                prod_weights = np.outer(samp_weights, samp_weights)
                samples[i, 0] = np.sum(prod_weights * below_rope)
                samples[i, 2] = np.sum(prod_weights * above_rope)
            samples[:, 1] = -samples[:, 0] - samples[:, 2] + 1
            return samples

        def without_rope():
            sums, weights = diff_sums(x, y)
            above_0 = heaviside(sums, 0)
            samples = np.zeros((nsamples, 3))
            _random_state = np.random.RandomState(random_state)
            for i, samp_weights in enumerate(_random_state.dirichlet(weights, nsamples)):
                prod_weights = np.outer(samp_weights, samp_weights)
                samples[i, 2] = np.sum(prod_weights * above_0)
            samples[:, 0] = -samples[:, 2] + 1
            return samples

        check_args(x, y, rope, nsamples)
        return with_rope() if rope > 0 else without_rope()


class HierarchicalTest(Test):
    """
    Compute a hierarchical t test.
    (`Statistical comparison of classifiers through Bayesian hierarchical
    modelling <http://repository.supsi.ch/7416/1/hierarchical.pdf>`_,
    G. Corani et al, Machine Learning, 2017).

    Arguments `x` and `y` should be two-dimensional arrays; rows correspond to
    data sets and elements within rows are scores obtained by (possibly
    repeated) cross-validation(s).
    """

    @classmethod
    # pylint: disable=arguments-differ
    # pylint: disable=unused-argument
    def sample(cls, x, y, rope, *, runs=1,
               lower_alpha=1, upper_alpha=2,
               lower_beta=0.01, upper_beta=0.1,
               upper_sigma=1000, chains=4, nsamples=10000,
               random_state=None):
        try:
            import stan
        except ImportError:
            raise ImportError("Hierarchical model requires 'pystan >= 3.4.0'; "
                              "install it e.g. by 'pip install pystan==3.4.0'")

        _random_state = np.random.RandomState(random_state)

        LAST_SAMPLE_PICKLE = "last-sample.pickle"
        stan_file = os.path.join(os.path.split(__file__)[0], "hierarchical-t-test.stan")

        args_signature = (
            hash(tuple(tuple(e for e in row) for row in y - x)),
            runs, lower_alpha, upper_alpha, lower_beta, upper_beta, upper_sigma)

        def try_unpickle(fname):
            if os.path.exists(fname) \
                    and os.path.getmtime(fname) > os.path.getmtime(stan_file):
                with open(fname, "rb") as f:
                    return pickle.load(f)
            return None

        def try_pickle(obj, fname):
            try:
                with open(fname, "wb") as f:
                    pickle.dump(obj, f)
            except PermissionError:
                pass

        def scaled_data(x, y, rope):
            # ensure homogenous scale across all data seta
            diff = y - x
            stds = np.std(diff, axis=1)
            std_diff = np.mean(stds)
            return rope / std_diff, diff / std_diff

        def prepare_stan_data(diff):
            ndatasets, nscores = diff.shape
            nfolds = nscores / runs

            # avoid numerical problems with zero variance
            nscores_2 = nscores // 2
            for sample in diff:
                if np.var(sample) == 0:
                    sample[:nscores_2] = _random_state.uniform(-rope, rope, nscores_2)
                    sample[nscores_2:] = -sample[:nscores_2]

            std_within = np.mean(np.std(diff, axis=1))  # may be different from std_diff!
            std_among = np.std(np.mean(diff, axis=1)) if ndatasets > 1 else std_within
            maxdiff = np.max(np.abs(diff))

            return dict(
                x=diff, Nsamples=nscores, q=ndatasets,
                rho=1 / nfolds,
                deltaLow=-maxdiff, deltaHi=maxdiff,
                lowerAlpha=lower_alpha, upperAlpha=upper_alpha,
                lowerBeta=lower_beta, upperBeta=upper_beta,
                stdLow=0, stdHi=std_within * upper_sigma,
                std0Low=0, std0Hi=std_among * upper_sigma
            )

        def run_stan(diff, **kwargs):
            stan_data = prepare_stan_data(diff)

            # check if the last pickled result can be reused
            cached = try_unpickle(LAST_SAMPLE_PICKLE)
            if cached is not None and cached[0] == args_signature:
                return cached[1]

            program_code = open(stan_file).read()
            model: stan.model.Model = stan.build(program_code, data=stan_data)
            fit: stan.fit.Fit = model.sample(**kwargs)

            mu = fit["delta0"][0]
            stdh = fit["std0"][0]
            nu = fit["nu"][0]
            try_pickle((args_signature, (mu, stdh, nu)), LAST_SAMPLE_PICKLE)
            return mu, stdh, nu

        rope, diff = scaled_data(x, y, rope)
        mu, stdh, nu = run_stan(diff, num_samples=nsamples, num_chains=chains)
        samples = np.empty((len(nu), 3))
        for mui, std, df, sample_row in zip(mu, stdh, nu, samples):
            sample_row[2] = 1 - stats.t.cdf(rope, df, mui, std)
            sample_row[0] = stats.t.cdf(-rope, df, mui, std)
            sample_row[1] = 1 - sample_row[0] - sample_row[2]
        return samples


def two_on_multiple(x, y, rope=0, *, runs=1, names=None, plot=False, **kwargs):
    """
    Compute probabilities using a Bayesian signed-ranks test (if `x` and `y` or
    one-dimensional) or a hierarchical (if they are two-dimensions), and,
    optionally, draw a histogram.

    The hierarchical test assumes that the classifiers were evaluated using
    cross validation; argument `runs` gives the number of repetitions of
    cross-validation.

    For more details, see :obj:`SignRankTest` and :obj:`HierarchicalTest`

    Args:
        x (np.array): a vector of scores for the first model
        y (np.array): a vector of scores for the second model
        rope (float): the width of the region of practical equivalence (default: 0)
        runs (int): the number of repetitions of cross validation
            (for hierarhical model) (default: 1)
        nsamples (int): the number of samples (default: 50000)
        plot (bool): if `True`, the function also return a histogram (default: False)
        names (tuple of str): names of classifiers (ignored if `plot` is `False`)
        random_state (int or None): random seed for drawing the samples, if None a random seed is picked

    Returns:
        `(p_left, p_rope, p_right)` if `rope > 0`; otherwise `(p_left, p_right)`.

        If `plot=True`, the function also returns a matplotlib figure,
        that is, `((p_left, p_rope, p_right), fig)`
        """
    if x.ndim == 2:
        test = HierarchicalTest
        kwargs["runs"] = runs
    else:
        test = SignedRankTest
    return call_shortcut(test, x, y, rope, names=names, plot=plot, **kwargs)
