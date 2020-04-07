import numpy as np


def get_Smean(f0_cntr):
    return np.mean(f0_cntr)


def get_Sdmean(f0_cntr):
    return np.mean(np.gradient(f0_cntr))


def get_Sstd(f0_cntr):
    return np.std(f0_cntr)


def get_Sdstd(f0_cntr):
    return np.std(np.gradient(f0_cntr))


def get_Srange(f0_cntr):
    return np.max(f0_cntr) - np.max(f0_cntr)


def get_Sdrange(f0_cntr):
    grad = np.gradient(f0_cntr)
    return np.max(grad) - np.min(grad)


def get_Smax(f0_cntr):
    return np.max(f0_cntr)


def get_Smin(f0_cntr):
    return np.min(f0_cntr)


def get_SQ25(f0_cntr):
    return np.quantile(f0_cntr, 0.25)


def get_SQ75(f0_cntr):
    return np.quantile(f0_cntr, 0.75)


def feat_ext(corpus, feats, save=None):
    samples = list()
    for i in range(len(corpus)):
        sample = list()
        for j in feats:
            sample.append(j(corpus[i]))
        samples.append(sample)

    samples = np.array(samples)
    if save is not None:
        with open(save, "wb") as f:
            pickle.dump(samples, f)

    return samples


feature_map = {
    "Smean": get_Smean,
    "Sdmean": get_Sdmean,
    "Sstd": get_Sstd,
    "Sdstd": get_Sdstd,
    "Srange": get_Srange,
    "Sdrange": get_Sdrange,
    "Smax": get_Smax,
    "Smin": get_Smin,
    "SQ25": get_SQ25,
    "SQ75": get_SQ75
}
