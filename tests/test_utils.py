import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from baycomp.utils import check_args, call_shortcut, seaborn_plt


class UtilsTest(unittest.TestCase):
    def test_check_args(self):
        y = x = np.ones((5, ))

        check_args(x, y)
        check_args(x, y, rope=10)
        check_args(x, y, rope=0)
        check_args(x, y, prior=11)
        check_args(x, y, prior=0)
        check_args(x, y, nsamples=1)
        check_args(x, y, nsamples=1.0)

        self.assertRaises(ValueError, check_args, np.ones((5, 1)), y)
        self.assertRaises(ValueError, check_args, x, np.ones((5, 1)))
        self.assertRaises(ValueError, check_args, x, np.ones((6, )))
        self.assertRaises(ValueError, check_args, x, y, rope=-1)
        self.assertRaises(ValueError, check_args, x, y, prior=-1)
        self.assertRaises(ValueError, check_args, x, y, nsamples=-1)
        self.assertRaises(ValueError, check_args, x, y, nsamples=3.14)

    def test_call_shortcut(self):
        y = x = np.ones((5, ))
        rope = 1
        names = ("a", "b")
        probs = (0.1, 0.2, 0.7)
        fig = object()

        sample = Mock(**{'probs.return_value': probs,
                         'plot.return_value': (fig, names)})
        test = Mock(return_value=sample)

        self.assertEqual(
            call_shortcut(test, x, y, rope, 42, names=names, foo=13),
            probs)
        test.assert_called_with(x, y, rope, 42, foo=13)

        self.assertEqual(
            call_shortcut(test, x, y, rope, 42, names=names, foo=13, plot=True),
            (probs, (fig, names)))
        test.assert_called_with(x, y, rope, 42, foo=13)


if __name__ == "__main__":
    unittest.main()
