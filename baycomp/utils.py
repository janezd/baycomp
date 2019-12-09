import contextlib


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


@contextlib.contextmanager
def seaborn_plt():
    # Set a Seaborn-like style. See https://github.com/mwaskom/seaborn.
    try:
        import matplotlib as mpl
    except ImportError:
        raise ImportError("Plotting requires 'matplotlib'; "
                          "use 'pip install matplotlib' to install it")

    params = {
        "font.size": 12,
        "text.color": ".15",
        "font.sans-serif": ['Arial', 'DejaVu Sans', 'Liberation Sans',
                            'Bitstream Vera Sans', 'sans-serif'],
        "legend.fontsize": 11,

        "axes.labelsize": 12,
        "axes.labelcolor": ".15",
        "axes.axisbelow": True,
        "axes.facecolor": "#EAEAF2",
        "axes.edgecolor": "white",
        "axes.linewidth": 1.25,

        "grid.linewidth": 1,
        "grid.color": "white",

        "xtick.labelsize": 11,
        "xtick.color": ".15",
        "xtick.major.width": 1.25,
        "ytick.left": False,

        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True}
    orig_params = {k: mpl.rcParams[k] for k in params}
    mpl.rcParams.update(params)

    converter = mpl.colors.colorConverter
    colors = dict(b="#4C72B0", g="#55A868", r="#C44E52",
                  m="#8172B3", y="#CCB974", c="#64B5CD",
                  k=(.1, .1, .1, .1))
    colors = {k: converter.to_rgb(v) for k, v in colors.items()}
    orig_colors = converter.colors.copy()
    orig_cache = converter.cache.copy()
    converter.colors.update(colors)
    converter.cache.update(colors)

    import matplotlib.pyplot as plt
    try:
        yield plt
    finally:
        mpl.rcParams.update(orig_params)
        converter.colors.clear()
        converter.colors.update(orig_colors)
        converter.cache.clear()
        converter.cache.update(orig_cache)
