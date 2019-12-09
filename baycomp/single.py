from functools import lru_cache

import numpy as np
from scipy import stats

from .utils import check_args, seaborn_plt, call_shortcut

__all__ = ["CorrelatedTTest", "two_on_single"]


class Posterior:
    """
    The posterior distribution of differences on a single data set.

    Args:
        mean (float): the mean difference
        var (float): the variance
        df (float): degrees of freedom
        rope (float): rope (default: 0)
        meanx (float): mean score of the first classifier; shown in a plot
        meany (float): mean score of the second classifier; shown in a plot
        names (tuple of str): names of classifiers; shown in a plot
        nsamples (int): the number of samples; used only in property `sample`,
            not in computation of probabilities or plotting (default: 50000)

        """
    def __init__(self, mean, var, df, rope=0, meanx=None, meany=None,
                 *, names=None, nsamples=50000):
        self.meanx = meanx
        self.meany = meany
        self.mean = mean
        self.var = var
        self.df = df
        self.rope = rope
        self.names = names
        self.nsamples = nsamples

    @property
    @lru_cache(1)
    def sample(self):
        """
        A sample of differences as 1-dimensional array.

        Like posteriors for comparison on multiple data sets, an instance of
        this class will always return the same sample.

        This sample is not used by other methods.
        """
        if self.var == 0:
            return np.full((self.nsamples, ), self.mean)
        return self.mean + \
               np.sqrt(self.var) * np.random.standard_t(self.df, self.nsamples)

    def probs(self):
        """
        Compute and return probabilities

        Probabilities are not computed from a sample posterior but
        from cumulative Student distribution.

        Returns:
            `(p_left, p_rope, p_right)` if `rope > 0`;
            otherwise `(p_left, p_right)`.
        """
        t_parameters = self.df, self.mean, np.sqrt(self.var)
        if self.rope == 0:
            if self.var == 0:
                pr = (self.mean > 0) + 0.5 * (self.mean == 0)
            else:
                pr = 1 - stats.t.cdf(0, *t_parameters)
            return 1 - pr, pr
        else:
            if self.var == 0:
                pl = float(self.mean < -self.rope)
                pr = float(self.mean > self.rope)
            else:
                pl = stats.t.cdf(-self.rope, *t_parameters)
                pr = 1 - stats.t.cdf(self.rope, *t_parameters)
        return pl, 1 - pl - pr, pr

    def plot(self, names=None):
        """
        Plot the posterior Student distribution as a histogram.

        Args:
            names (tuple of str): names of classifiers

        Returns:
            matplotlib figure
        """
        with seaborn_plt() as plt:
            names = names or self.names or ("C1", "C2")

            fig, ax = plt.subplots()
            ax.grid(True)
            label = "difference"
            if self.meanx is not None and self.meany is not None:
                label += " ({}: {:.3f}, {}: {:.3f})".format(
                    names[0], self.meanx, names[1], self.meany)
            ax.set_xlabel(label)
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


class CorrelatedTTest:
    """
    Compute and plot a Bayesian correlated t-test
    """
    def __new__(cls, x, y, rope=0, runs=1, *, names=None, nsamples=50000):
        check_args(x, y, rope)
        if not int(runs) == runs > 0:
            raise ValueError('Number of runs must be a positive integer')
        if len(x) % round(runs) != 0:
            raise ValueError("Number of measurements is not divisible by number of runs")

        mean, var, df = cls.compute_statistics(x, y, runs)
        return Posterior(mean, var, df, rope,
                         np.mean(x), np.mean(y), names=names,
                         nsamples=nsamples)

    @classmethod
    def compute_statistics(cls, x, y, runs=1):
        """
        Compute statistics (mean, variance) from the differences.

        The number of runs is needed to compute the Nadeau-Bengio correction
        for underestimated variance.

        Args:
            x (np.array): a vector of scores for the first model
            y (np.array): a vector of scores for the second model
            runs (int): number of repetitions of cross validation (default: 1)

        Returns:
            mean, var, degrees_of_freedom
        """
        diff = y - x
        n = len(diff)
        nfolds = n / runs
        mean = np.mean(diff)
        var = np.var(diff, ddof=1)
        var *= 1 / n + 1 / (nfolds - 1)  # Nadeau-Bengio's correction
        return mean, var, n - 1

    @classmethod
    def sample(cls, x, y, runs=1, *, nsamples=50000):
        """
        Return a sample of posterior distribution for the given data

        Args:
            x (np.array): a vector of scores for the first model
            y (np.array): a vector of scores for the second model
            runs (int): number of repetitions of cross validation (default: 1)
            nsamples (int): the number of samples (default: 50000)

        Returns:
            mean, var, degrees_of_freedom
        """
        return cls(x, y, runs=runs, nsamples=nsamples).sample

    @classmethod
    def probs(cls, x, y, rope=0, runs=1):
        """
        Compute and return probabilities

        Probabilities are not computed from a sample posterior but
        from cumulative Student distribution.

        Args:
            x (np.array): a vector of scores for the first model
            y (np.array): a vector of scores for the second model
            rope (float): the width of the region of practical equivalence (default: 0)
            runs (int): number of repetitions of cross validation (default: 1)

        Returns:
            `(p_left, p_rope, p_right)` if `rope > 0`;
            otherwise `(p_left, p_right)`.
        """
        # new returns an instance of Test, not CorrelatedTTest
        # pylint: disable=no-value-for-parameter
        return cls(x, y, rope, runs).probs()

    @classmethod
    def plot(cls, x, y, rope=0, runs=1, *, names=None):
        """
        Plot the posterior Student distribution as a histogram.

        Args:
            x (np.array): a vector of scores for the first model
            y (np.array): a vector of scores for the second model
            rope (float): the width of the region of practical equivalence (default: 0)
            names (tuple of str): names of classifiers

        Returns:
            matplotlib figure
        """
        # new returns an instance of Test, not CorrelatedTTest
        # pylint: disable=no-value-for-parameter
        return cls(x, y, rope, runs).plot(names)


def two_on_single(x, y, rope=0, runs=1, *, names=None, plot=False):
    """
    Compute probabilities using a Bayesian correlated t-test and,
    optionally, draw a histogram.

    The test assumes that the classifiers were evaluated using cross
    validation. Argument `runs` gives the number of repetitions of
    cross-validation.

    For more details, see :obj:`CorrelatedTTest`

    Args:
        x (np.array): a vector of scores for the first model
        y (np.array): a vector of scores for the second model
        rope (float): the width of the region of practical equivalence (default: 0)
        runs (int): the number of repetitions of cross validation (default: 1)
        nsamples (int): the number of samples (default: 50000)
        plot (bool): if `True`, the function also return a histogram (default: False)
        names (tuple of str): names of classifiers (ignored if `plot` is `False`)

    Returns:
        `(p_left, p_rope, p_right)` if `rope > 0`; otherwise `(p_left, p_right)`.

        If `plot=True`, the function also returns a matplotlib figure,
        that is, `((p_left, p_rope, p_right), fig)`
        """
    return call_shortcut(CorrelatedTTest, x, y, rope,
                         plot=plot, names=names, runs=runs)
