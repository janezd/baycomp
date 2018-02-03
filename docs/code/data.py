import os
from functools import lru_cache

import numpy as np


@lru_cache(1)
def _read_data():
    data = []
    datasets = []
    classifiers = []
    basedir = os.path.split(__file__)[0]
    with open(os.path.join(basedir, "accuracies.txt")) as f:
        classifier = f.readline().strip()
        while True:  # loop over classifier
            if not classifier:
                break
            classifiers.append(classifier)
            data.append([])
            t_datasets = datasets and []
            while True:  # loop over data sets
                line = f.readline().strip()
                dataset, *scores = line.split() if line else ("",)
                if not scores:
                    # Check that order of data sets is same for all classifiers
                    assert datasets == t_datasets
                    classifier = dataset
                    break
                data[-1].append([float(x) for x in scores])
                t_datasets.append(dataset)
    return np.array(data), classifiers, datasets


def get_data(classifier=..., dataset=..., aggregate=False):
    def get_indices(names, pool):
        if names is ...:
            return np.arange(len(pool), dtype=int)
        if isinstance(names, str):
            return np.array([pool.index(names)])
        else:
            return np.array([pool.index(name) for name in names])

    data, classifiers, datasets = _read_data()
    data = data[np.ix_(get_indices(classifier, classifiers),
                       get_indices(dataset, datasets))]
    if aggregate:
        data = np.mean(data, axis=2)
    data = data.squeeze()
    return data


def get_classifiers():
    return _read_data()[1]


def get_datasets():
    return _read_data()[2]
