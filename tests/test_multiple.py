import unittest
from unittest.mock import patch, Mock

import numpy as np

from tests.utils import TestTestBase
from baycomp.multiple import \
    SignTest, SignedRankTest, HierarchicalTest, two_on_multiple, \
    Test, Posterior


class PosteriorTest(unittest.TestCase):
    def test_probs(self):
        sample = np.array([[0.2, 0.1, 0.7],
                           [0.4, 0.1, 0.5],
                           [0.1, 0.1, 0.8],
                           [0.9, 0.0, 0.1],
                           [0.8, 0.1, 0.1],
                           [0, 1, 0]])
        posterior = Posterior(sample)
        np.testing.assert_almost_equal(
            np.array(posterior.probs()), np.array([2, 1, 3]) / 6)
        np.testing.assert_almost_equal(
            np.array(posterior.probs(True)), np.array([2, 1, 3]) / 6)
        np.testing.assert_almost_equal(
            np.array(posterior.probs(False)), np.array([2, 3]) / 5)

    @patch.object(Posterior, "plot_histogram")
    @patch.object(Posterior, "plot_simplex")
    def test_plot(self, mocksimplex, mockhistogram):
        names = ("a", "b")
        names2 = ("c", "d")

        sample = np.array([[0.2, 0.2, 0.6], [0.1, 0, 0.9]])
        posterior = Posterior(sample, names=names)
        posterior.plot()
        mocksimplex.assert_called_with(None)
        mockhistogram.assert_not_called()

        posterior.plot(names2)
        mocksimplex.assert_called_with(names2)
        mockhistogram.assert_not_called()

        mocksimplex.reset_mock()
        sample = np.array([[0.35, 0.05, 0.6], [0.1, 0, 0.9]])
        posterior = Posterior(sample, names=names)
        posterior.plot()
        mockhistogram.assert_called_with(None)
        mocksimplex.assert_not_called()

        posterior.plot(names2)
        mockhistogram.assert_called_with(names2)
        mocksimplex.assert_not_called()


class TestTest(TestTestBase):
    @patch.object(Test, "sample", return_value=Mock())
    @patch("baycomp.multiple.Posterior")
    def test_new(self, mockposterior, mocksample):
        x = np.array([4, 2, 6])
        y = np.array([5, 7, 3])
        Test(x, y, 1, nsamples=3, foo=42)
        mocksample.assert_called_with(x, y, 1, nsamples=3, foo=42)
        mockposterior.assert_called_with(mocksample.return_value)

    def test_probs(self):
        x = np.array([4, 2, 6])
        y = np.array([5, 7, 3])

        self.assert_forwards(
            Test, "probs", x, y, 0.5, nsamples=42,
            new_args=(x, y, 0.5), new_kwargs=dict(nsamples=42),
            meth_args=(True, )
        )
        self.assert_forwards(
            Test, "probs", x, y, 0, nsamples=42,
            new_args=(x, y, 0), new_kwargs=dict(nsamples=42),
            meth_args=(False, )
        )
        self.assert_forwards(
            Test, "probs", x, y, nsamples=42,
            new_args=(x, y, 0), new_kwargs=dict(nsamples=42),
            meth_args=(False, )
        )

    def test_plot(self):
        x = np.array([4, 2, 6])
        y = np.array([5, 7, 3])
        names = ("a", "b")

        self.assert_forwards(
            Test, "plot", x, y, 0.5, nsamples=42, names=names,
            new_args=(x, y, 0.5), new_kwargs=dict(nsamples=42),
            meth_args=(names, )
        )

        self.assert_forwards(
            Test, "plot_simplex", x, y, 0.5, nsamples=42, names=names,
            new_args=(x, y, 0.5), new_kwargs=dict(nsamples=42),
            meth_args=(names, )
        )

        self.assert_forwards(
            Test, "plot_histogram", x, y, nsamples=42, names=names,
            new_args=(x, y), new_kwargs=dict(nsamples=42, rope=0),
            meth_args=(names, )
        )


class SignTestTest(unittest.TestCase):
    @patch("numpy.random.dirichlet")
    def test_sample(self, mockdirichlet):
        x = np.array([15, 16, 17, 24, 11, 12, 13, 14])
        y = np.array([10, 13, 15, 24, 15, 82, 83, 84])
        # diff =      -5  -3  -2   0  4   70  70  70

        def assert_dirichlet(s, nsamples=50000):
            alpha, ns = mockdirichlet.call_args[0]
            np.testing.assert_almost_equal(alpha, np.array(s) + 0.0001)
            self.assertEqual(ns, nsamples)

        SignTest.sample(x, y, prior=0)
        assert_dirichlet([3, 1, 4])
        SignTest.sample(x, y, prior=0, rope=1)
        assert_dirichlet([3, 1, 4])
        SignTest.sample(x, y, prior=0, rope=2)
        assert_dirichlet([2, 2, 4])
        SignTest.sample(x, y, prior=0, rope=3)
        assert_dirichlet([1, 3, 4])
        SignTest.sample(x, y, prior=0, rope=4)
        assert_dirichlet([1, 4, 3])
        SignTest.sample(x, y, prior=0, rope=5)
        assert_dirichlet([0, 5, 3])
        SignTest.sample(x, y, prior=0, rope=100)
        assert_dirichlet([0, 8, 0])

        SignTest.sample(x, y, prior=0.5, rope=1)
        assert_dirichlet([3, 1.5, 4])
        SignTest.sample(x, y, prior=(0.5, SignTest.LEFT), rope=1)
        assert_dirichlet([3.5, 1, 4])
        SignTest.sample(x, y, prior=(0.5, SignTest.ROPE), rope=1)
        assert_dirichlet([3, 1.5, 4])
        SignTest.sample(x, y, prior=(0.5, SignTest.RIGHT), rope=1)
        assert_dirichlet([3, 1, 4.5])

        SignTest.sample(x, y, prior=0, nsamples=42)
        assert_dirichlet([3, 1, 4], 42)

    @patch("baycomp.multiple.check_args")
    @patch("numpy.random.dirichlet")
    def test_sample_checks_args(self, _, mockcheckargs):
        x = np.array([15, 16, 17, 24, 11, 12, 13, 14])
        y = np.array([10, 13, 15, 24, 15, 82, 83, 84])
        SignTest.sample(x, y, rope=1, prior=0.5, nsamples=42)
        mockcheckargs.assert_called_with(x, y, 1, 0.5, 42)


class SignedRankTestTest(unittest.TestCase):
    def test_sample(self):
        # TODO This tests only that sampling does not crash
        x = np.array([15, 16, 17, 24, 11, 12, 13, 14])
        y = np.array([10, 13, 15, 24, 15, 82, 83, 84])

        def test(*args, **kwargs):
            self.assertEqual(
                SignedRankTest.sample(*args, nsamples=10, **kwargs).shape,
                (10, 3))

        test(x, y)
        test(x, y, rope=1)
        test(x, y, rope=3)
        test(x, y, rope=100)
        test(x, y, prior=5)
        test(x, y, prior=5, rope=10)


class TwoOnMultipleTest(unittest.TestCase):
    @patch("baycomp.multiple.call_shortcut")
    def test_two_on_multiple(self, mockcall):
        names = ("a", "b")

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        two_on_multiple(x, y, 0.5, runs=10, plot=True, names=names)
        mockcall.assert_called_with(
            SignedRankTest, x, y, 0.5, plot=True, names=names)

        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([[4, 5, 6], [1, 2, 3]])
        two_on_multiple(x, y, 0.5, plot=True, names=names)
        mockcall.assert_called_with(
            HierarchicalTest, x, y, 0.5, plot=True, runs=1, names=names)

        two_on_multiple(x, y, 0.5, runs=10, plot=True, names=names)
        mockcall.assert_called_with(
            HierarchicalTest, x, y, 0.5, plot=True, runs=10, names=names)
