import unittest
from unittest.mock import patch, Mock


class TestTestBase(unittest.TestCase):
    def assert_forwards(self, class_, method_name, *args,
                        new_args=None, new_kwargs=None,
                        meth_args=None, meth_kwargs=None, **kwargs):
        def default(x, value):
            return x if x is not None else value
        new_args = default(new_args, args)
        new_kwargs = default(new_kwargs, kwargs)
        meth_args = default(meth_args, ())
        meth_kwargs = default(meth_kwargs, {})

        mockres = object()
        mockmethod = Mock(return_value=mockres)
        with patch.object(
                class_, "__new__",
                return_value=Mock(**{method_name: mockmethod})) as mocknew:
            method = getattr(class_, method_name)
            self.assertEqual(method(*args, **kwargs), mockres)
            mocknew.assert_called_with(class_, *new_args, **new_kwargs)
            mockmethod.assert_called_with(*meth_args, **meth_kwargs)



