import matplotlib.pyplot as plt

import baycomp as bc
from data import get_data

data_nbc = get_data("nbc", "squash-unstored")
data_aode = get_data("aode", "squash-unstored")

data_nbc = get_data("nbc", aggregate=True)
data_aode = get_data("aode", aggregate=True)
data_j48 = get_data("j48", aggregate=True)


t = bc.SignTest(data_nbc, data_aode, 1)
print(t.p_values())
t.plot()

print(bc.SignTest.test(data_nbc, data_aode, 1))
bc.SignTest.plot(data_nbc, data_aode, 1)

data_nbc = get_data("nbc")
data_aode = get_data("aode")
print(bc.two_on_multiple(data_nbc, data_aode, 0.1, rho=0.1, hierarchical=True))
sample = bc.HierarchicalTest.sample(data_nbc, data_aode, 0.3, rho=0.1)
bc.HierarchicalTest.plot(data_nbc, data_aode, 0.3, rho=0.1, names=("nbc", "aode"))

sample = bc.HierarchicalTest(data_nbc, data_aode, 0.3, rho=0.1)
sample.plot(names=("nbc", "aode"))
print(sample.p_values())

plt.show()
