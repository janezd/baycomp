import matplotlib.pyplot as plt

import baycomp
from data import get_data

data_nbc = get_data("nbc", "squash-unstored")
data_aode = get_data("aode", "squash-unstored")
#baycomp.plot_posterior_t(data_nbc, data_aode, rope=1, runs=10, names=("nbc", "aode"))

data_nbc = get_data("nbc", aggregate=True)
data_aode = get_data("aode", aggregate=True)
data_j48 = get_data("j48", aggregate=True)
#baycomp.plot_posterior_sign_rank(data_nbc, data_aode, 1, names=("nbc", "aode"))
baycomp.plot_posterior_sign(data_nbc, data_aode, 0.2, names=("nbc", "aode"))

#print(baycomp.signtest(data_nbc, data_aode, 0.5))
#print(baycomp.signranktest(data_nbc, data_aode, 0))
#print(baycomp.signranktest(data_nbc, data_aode, 1))

data_nbc = get_data("nbc")
data_aode = get_data("aode")
print(baycomp.hierarchical_t(data_nbc, data_aode, 0.2))
ps, by_data_sets, sample = baycomp.hierarchical_t(data_nbc, data_aode, 0.2, verbose_result=True)
baycomp.plot_simplex(sample)

plt.show()
