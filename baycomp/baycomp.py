import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    plt = None

try:
    import seaborn as sns
    sns.set(color_codes=True)
except ImportError:
    sns = None


__all__ = ["two_on_single", "two_on_multiple",
           "signtest", "plot_posterior_sign",
           "signranktest", "plot_posterior_sign_rank",
           "correlated_t", "plot_posterior_t",
           "LEFT", "ROPE", "RIGHT",
           "plot_simplex"]


def requires(module, library):
    def wrapper(f):
        def check_and_call(*args, **kwargs):
            if module is None:
                raise ImportError(
                    "'{0}' requires '{1}'; "
                    "use 'pip install {1}' to install it"
                    .format(f.__name__, library))
            return f(*args, **kwargs)
        return check_and_call
    return wrapper


def _check_args(x, y, rope=0, prior=1, nsamples=50000, runs=1):
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
    if not round(runs) == runs > 0:
        raise ValueError("Number of runs must be a positive integer")
    if len(x) % round(runs) != 0:
        raise ValueError("Number of measurements is not divisible by number of runs")


def _compute_posterior_t_parameters(x, y, runs=1):
    if not int(runs) == runs > 0:
        raise ValueError('Number of runs must be a positive integer')
    diff = y - x
    n = len(diff)
    nfolds = n / runs
    mean = np.mean(diff)
    var = np.var(diff, ddof=1)
    var *= 1 / n + 1 / (nfolds - 1)  # Nadeau-Bengio's correction
    return mean, var, n - 1


def correlated_t(x, y, rope=0, runs=1):
    """
    Compute correlated t-test

    The function uses the Bayesian interpretation of the p-value and returns
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

    Returns:
        (p_left, p_rope, p_right), or (p_left, p_right) if rope is 0
    """
    _check_args(x, y, rope, runs=runs)
    mean, var, df = _compute_posterior_t_parameters(x, y, runs)

    if var == 0:
        return float(mean < rope), float(-rope <= mean <= rope), float(rope < mean)
    greater = stats.t.cdf(mean, df, rope, np.sqrt(var))
    less = 1 - stats.t.cdf(mean, df, -rope, np.sqrt(var))
    if rope == 0:
        return less, greater
    return less, 1 - greater - less, greater


@requires(plt, "matplotlib")
def plot_posterior_t(x, y, rope=0.1, runs=1, names=None):
    """
    Plot posterior distribution and rope intervals for correlated t-test

    See documentation for function :obj:`correlated_t` for details.

    Args:
        x (np.array): a vector of scores for the first model
        y (np.array): a vector of scores for the second model
        rope (float): the width of the rope (default: 0)
        runs (int): number of repetitions of cross validation (default: 1)
        names (pair of str): the names of the two classifiers (default: None,
            no names are shown on the plot)

    Returns:
        matplotlib.pyplot.figure
    """
    _check_args(x, y, rope, runs=runs)
    mean, var, df = _compute_posterior_t_parameters(x, y, runs)

    fig, ax = plt.subplots()
    ax.grid(True)
    x_label = "difference"
    if names is not None:
        x_label += " ({}: {:.3f}, {}: {:.3f})".format(
            names[0], np.mean(x), names[1], np.mean(y))
    ax.set_xlabel(x_label)
    ax.get_yaxis().set_ticklabels([])
    ax.axvline(x=-rope, color="#ffad2f", linewidth=2, label="rope")
    ax.axvline(x=rope, color="#ffad2f", linewidth=2)

    targs = (df, mean, np.sqrt(var))
    xs = np.linspace(min(stats.t.ppf(0.005, *targs), -1.05 * rope),
                     max(stats.t.ppf(0.995, *targs), 1.05 * rope),
                     100)
    ys = stats.t.pdf(xs, *targs)
    ax.plot(xs, ys, color="#2f56e0", linewidth=2, label="pdf")
    ax.fill_between(xs, ys, np.zeros(100), color="#34ccff")
    ax.legend()
    return fig


def p_values(samples, with_rope):
    # TODO: we ignore ties here; inconsequential?
    winners = np.argmax(samples, axis=1)
    pl, pe, pr = np.bincount(winners, minlength=3) / len(winners)
    if with_rope:
        return pl, pe, pr
    return pl, pr

LEFT, ROPE, RIGHT = range(3)

def monte_carlo_samples_sign(x, y, rope=0, prior=1, nsamples=50000):
    if isinstance(prior, tuple):
        prior, prior_place = prior
    else:
        prior_place = ROPE
    _check_args(x, y, rope, prior, nsamples)

    diff = y - x
    nleft = sum(diff < -rope)
    nright = sum(diff > rope)
    nrope = len(diff) - nleft - nright
    alpha = np.array([nleft, nrope, nright], dtype=float)
    alpha += 0.0001  # for numerical stability
    alpha[prior_place] += prior
    return np.random.dirichlet(alpha, nsamples)  # pylint: disable=no-member


def signtest(x, y, rope=0, prior=1, nsamples=50000):
    """
    Args:
        x (np.array): a vector of scores for the first model
        y (np.array): a vector of scores for the second model
        rope (float): the width of the rope (default: 0)
        prior (float or tuple): prior strength, or tuple with strength and
            place (`LEFT`, `ROPE` or `RIGHT`); (default: 1)
        nsamples (int): the number of Monte Carlo samples (default: 50000)

    Returns:
        (p_left, p_rope, p_right) if rope > 0, else (p_left, p_right)
    """
    samples = monte_carlo_samples_sign(x, y, rope, prior, nsamples)
    return p_values(samples, rope > 0)


def heaviside(a, thresh):
    return (a > thresh).astype(float) + (a == thresh).astype(float) * 0.5


def diff_sums(x, y):
    diff = np.hstack(([0], y - x))
    diff_m = np.lib.stride_tricks.as_strided(
        diff, strides=diff.strides + (0,), shape=diff.shape * 2)
    weights = np.ones(len(diff))
    weights[0] = 0.5
    return diff_m + diff_m.T, weights


def monte_carlo_samples_rank(x, y, rope, nsamples=50000):
    def with_rope():
        sums, weights = diff_sums(x, y)
        above_rope = heaviside(sums, 2 * rope)
        below_rope = heaviside(-sums, 2 * rope)
        samples = np.zeros((nsamples, 3))
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
        for i, samp_weights in enumerate(np.random.dirichlet(weights, nsamples)):
            prod_weights = np.outer(samp_weights, samp_weights)
            samples[i, 2] = np.sum(prod_weights * above_0)
        samples[:, 0] = -samples[:, 2] + 1
        return samples

    _check_args(x, y, rope, nsamples)
    return with_rope() if rope > 0 else without_rope()


def signranktest(x, y, rope=0, nsamples=50000):
    """
    Args:
        x (np.array): a vector of scores for the first model
        y (np.array): a vector of scores for the second model
        rope (float): the width of the rope (default: 0)
        nsamples (int): the number of Monte Carlo samples (default: 50000)

    Returns:
        (p_left, p_rope, p_right) if rope > 0, else (p_left, p_right)
    """

    samples = monte_carlo_samples_rank(x, y, rope, nsamples)
    return p_values(samples, rope > 0)


@requires(plt, "matplotlib")
def plot_posterior_sign(x, y, rope=0, prior=1, nsamples=50000,
                        names=('C1', 'C2')):
    """
    Args:
        x (np.array): a vector of scores for the first model
        y (np.array): a vector of scores for the second model
        rope (float): the width of the rope (default: 0)
        prior (float or tuple): prior strength, or tuple with strength and
            place (`LEFT`, `ROPE` or `RIGHT`); (default: 1)
        nsamples (int): the number of Monte Carlo samples (default: 50000)
        names (pair of str): the names of the two classifiers (default: C1, C2)

    Returns:
        matplotlib.pyplot.figure
    """
    points = monte_carlo_samples_sign(x, y, rope, prior, nsamples)
    return plot_simplex(points, names)


@requires(plt, "matplotlib")
def plot_posterior_sign_rank(x, y, rope=0, nsamples=50000, names=('C1', 'C2')):
    """
    Args:
        x (np.array): a vector of scores for the first model
        y (np.array): a vector of scores for the second model
        rope (float): the width of the rope (default: 0)
        nsamples (int): the number of Monte Carlo samples (default: 50000)
        names (pair of str): the names of the two classifiers (default: C1, C2)

    Returns:
        matplotlib.pyplot.figure
    """
    points = monte_carlo_samples_rank(x, y, rope, nsamples)
    return plot_simplex(points, names)


@requires(plt, "matplotlib")
def plot_simplex(points, names=('C1', 'C2')):
    def _project(points):
        from math import sqrt, sin, cos, pi
        p1, p2, p3 = points.T / sqrt(3)
        x = (p2 - p1) * cos(pi / 6) + 0.5
        y = p3 - (p1 + p2) * sin(pi / 6) + 1 / (2 * sqrt(3))
        return np.vstack((x, y)).T

    pl, pe, pr = p_values(points, True)
    if pe == 0:
        return plot2d(points[:, 2], pl, pr, names)

    vert0 = _project(np.array(
        [[0.3333, 0.3333, 0.3333], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]))

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
    tripts = _project(points[:, [0, 2, 1]])
    # pylint: disable=no-member
    plt.hexbin(tripts[:, 0], tripts[:, 1], mincnt=1, cmap=plt.cm.Blues_r)
    # Leave some padding around the triangle for vertex labels
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')
    return fig


@requires(plt, "matplotlib")
def plot2d(points, pl, pr, names):
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.hist(points, 50, color="#34ccff")
    ax.axis(xmin=0, xmax=1)
    ax.text(0, 0, "\np({}) = {:.3f}".format(names[0], pl),
            horizontalalignment='left', verticalalignment='top')
    ax.text(1, 0, "\np({}) = {:.3f}".format(names[1], pr),
            horizontalalignment='right', verticalalignment='top')
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.axvline(x=0.5, color="#ffad2f", linewidth=2)
    return fig


two_on_single = correlated_t
two_on_multiple = signtest
