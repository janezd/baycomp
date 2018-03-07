try:
    import seaborn
    seaborn.set(color_codes=True)
# Can be import error or matplotlib's complaints about macOS platform
except Exception:  # pylint: disable=broad-except
    pass


def import_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("Plotting requires 'matplotlib'; "
                          "use 'pip install matplotlib' to install it")


def check_args(x, y, rope=0, prior=1, nsamples=50000):
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


def call_shortcut(test, x, y, rope, *args, plot=False, names=None, **kwargs):
    sample = test(x, y, rope, *args, **kwargs)
    if plot:
        return sample.probs(), sample.plot(names)
    else:
        return sample.probs()
