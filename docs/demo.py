import matplotlib.pyplot as plt

import baycomp
from data import get_data

data_nbc = get_data("nbc", "squash-unstored")
data_aode = get_data("aode", "squash-unstored")
baycomp.plot_posterior_t(data_nbc, data_aode, rope=1, runs=10, names=("nbc", "aode"))

data_nbc = get_data("nbc", aggregate=True)
data_aode = get_data("aode", aggregate=True)
baycomp.plot_posterior_sign(data_nbc, data_aode, 1, names=("nbc", "aode"))

plt.show()
