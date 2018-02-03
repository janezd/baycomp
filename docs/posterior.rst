.. currentmodule:: baycomp


Querying posterior distributions
================================

The third way to use the library is to construct and query posterior
distributions.

We construct the posterior distribution by calling the corresponding test
class. If `j48` and `nbc` contain scores from cross validation on a single
data set, we construct the posterior by

    >>> posterior = CorrelatedTTest(nbc, j48)

and then compute the probabilities and plot the histogram

    >>> posterior.probs()
    (0.4145119975061462, 0.5854880024938538)
    >>> fig = posterior.plot(names=("nbc", "j48"))

For comparison on multiple data sets we do the same, except that `nbc` and
`j48` must contain average classification accuracies (for sign test and
signed rank test) or a matrix of accuracies (for hierarchical test).

    >>> posterior = SignedRankTest(nbc, j48, rope=1)
    >>> posterior.probs()
    (0.23014, 0.00674, 0.76312)
    >>> fig = posterior.plot(names=("nbc", "j48"))

Single data set
---------------

.. autoclass:: baycomp.single.Posterior
    :members:

    Unlike the posterior for comparisons on multiple data sets, this
    distribution is not sampled; probabilities are computed from the
    posterior Student distribution.

    The class can provide a sample (as 1-dimensional array), but the
    sample itself is not used by other methods.


Multiple data sets
------------------

.. autoclass:: baycomp.multiple.Posterior
    :members:
