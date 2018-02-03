import unittest
from unittest.mock import patch

import numpy as np

from tests.utils import TestTestBase
from baycomp.single import CorrelatedTTest, Posterior, two_on_single


class PosteriorTest(unittest.TestCase):
    def test_sample(self):
        posterior = Posterior(42, 0, 1, nsamples=10)
        np.testing.assert_almost_equal(posterior.sample, np.full((10, ), 42))

        posterior = Posterior(42, 2, 3, nsamples=10)
        def mockstandardt(df, nsamples):
            return np.arange(nsamples) * df

        with patch("numpy.random.standard_t", mockstandardt):
            np.testing.assert_almost_equal(
                    posterior.sample,
                    42 + np.sqrt(2) * np.arange(10) * 3)

    def test_probs_var_0(self):
        # var=0, df=1, rope=2
        np.testing.assert_equal(Posterior(42, 0, 1, 2).probs(), [0, 0, 1])
        np.testing.assert_equal(Posterior(2, 0, 1, 2).probs(), [0, 1, 0])
        np.testing.assert_equal(Posterior(.5, 0, 1, 2).probs(), [0, 1, 0])
        np.testing.assert_equal(Posterior(0, 0, 1, 2).probs(), [0, 1, 0])
        np.testing.assert_equal(Posterior(-.5, 0, 1, 2).probs(), [0, 1, 0])
        np.testing.assert_equal(Posterior(-2, 0, 1, 2).probs(), [0, 1, 0])
        np.testing.assert_equal(Posterior(-42, 0, 1, 2).probs(), [1, 0, 0])

        # var=0, df=1, rope=0
        np.testing.assert_equal(Posterior(42, 0, 1).probs(), [0, 1])
        np.testing.assert_equal(Posterior(0, 0, 1).probs(), [0.5, 0.5])
        np.testing.assert_equal(Posterior(-42, 0, 1).probs(), [1, 0])

    def test_probs(self):
        def mockcdf(x, a_df, a_loc, a_scale):
            self.assertEqual(a_df, 1)
            self.assertEqual(a_loc, 3)
            self.assertAlmostEqual(a_scale, 5)
            # 0.25 below rope (or 0)
            # 0.4 below + between
            return 0.25 if x <= 0 else 0.4

        with patch("scipy.stats.t.cdf", mockcdf):
            posterior = Posterior(mean=3, var=25, df=1, rope=0)
            np.testing.assert_almost_equal(
                posterior.probs(), [0.25, 0.75])

            posterior = Posterior(mean=3, var=25, df=1, rope=2)
            np.testing.assert_almost_equal(
                posterior.probs(), [0.25, 0.15, 0.6])


class CorrelatedTTestTest(TestTestBase):
    def test_new(self):
        x = np.array([4, 2, 6])
        y = np.array([5, 7, 3])
        with patch.object(CorrelatedTTest,
                          "compute_statistics", return_value=(1, 2, 3)) as cs:
            posterior = CorrelatedTTest(x, y, rope=6, runs=3, names=("a", "b"))
            cs.assert_called_with(x, y, 3)
            self.assertEqual(posterior.mean, 1)
            self.assertEqual(posterior.var, 2)
            self.assertEqual(posterior.df, 3)
            self.assertEqual(posterior.rope, 6)
            self.assertEqual(posterior.meanx, 4)
            self.assertEqual(posterior.meany, 5)
            self.assertEqual(posterior.names, ("a", "b"))

    def test_new_checks_errors(self):
        x = np.array([4, 2, 6])
        y = np.array([5, 7, 3])
        with patch("baycomp.single.check_args") as ca:
            CorrelatedTTest(x, y, 12)
            ca.assert_called_with(x, y, 12)
        self.assertRaises(ValueError, CorrelatedTTest, x, y, runs=0)
        self.assertRaises(ValueError, CorrelatedTTest, x, y, runs=2)
        self.assertRaises(ValueError, CorrelatedTTest, x, y, runs=-1)
        self.assertRaises(ValueError, CorrelatedTTest, x, y, runs=0.5)

    def test_compute_statistics(self):
        x = np.array([4, 2, 6])
        y = np.array([5, 7, 3])
        self.assertEqual(
            CorrelatedTTest.compute_statistics(x, y, 1),
            (1, (4 ** 2 + 4 ** 2) / 2 * (1 / 3 + 1 / 2), 2))

        x = np.array([4, 2, 6, 4])
        y = np.array([5, 7, 3, 5])
        self.assertEqual(
            CorrelatedTTest.compute_statistics(x, y, 2),
            (1, (4 ** 2 + 4 ** 2) / 3 * (1 / 4 + 1), 3))

        x = np.ones(10)
        y = np.zeros(10)
        self.assertEqual(
            CorrelatedTTest.compute_statistics(x, y),
            (-1, 0, 9))

    def test_sample(self):
        x = np.array([42, 42, 42])
        y = np.zeros(3)
        np.testing.assert_almost_equal(
            CorrelatedTTest.sample(x, y, 1, nsamples=10),
            np.full((10, ), -42))

        def mockstandardt(df, nsamples):
            return np.arange(nsamples) * df

        with patch("numpy.random.standard_t", mockstandardt),\
                patch.object(CorrelatedTTest, "compute_statistics",
                             return_value=(1, 2, 3)):
            np.testing.assert_almost_equal(
                CorrelatedTTest.sample(x, y, 1, nsamples=10),
                1 + np.sqrt(2) * np.arange(10) * 3)

    def test_probs(self):
        x, y = object(), object()
        self.assert_forwards(CorrelatedTTest, "probs", x, y, 2, 1)

    def test_plot(self):
        x, y = object(), object()
        names = object()
        self.assert_forwards(
            CorrelatedTTest, "plot", x, y, 2, 1, names=names,
            new_args=(x, y, 2, 1), new_kwargs={}, meth_args=(names,))


class TwoOnSingleTest(unittest.TestCase):
    def test_two_on_single(self):
        x, y = object(), object()
        names = ("a, b")
        with patch("baycomp.single.call_shortcut") as mockshortcut:
            two_on_single(x, y, 0.5, 10, plot=True, names=names)
            mockshortcut.assert_called_with(CorrelatedTTest, x, y, 0.5,
                                            plot=True, names=names, runs=10)


if __name__ == "__main__":
    unittest.main()
