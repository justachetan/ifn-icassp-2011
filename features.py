import numpy as np
import scipy.stats as stats


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


def get_Smedian(f0_cntr):
    return np.median(f0_cntr)


def get_Sdmedian(f0_cntr):
    grad = np.gradient(f0_cntr)
    return np.median(grad)


def get_Siqr(f0_cntr):
    uqr = get_SQ75(f0_cntr)
    lqr = get_SQ25(f0_cntr)
    return uqr - lqr


def get_Sdiqr(f0_cntr):
    grad = np.gradient(f0_cntr)
    return get_Siqr(grad)


def get_Skurt(f0_cntr):
    return stats.kurtosis(f0_cntr)


def get_Sdkurt(f0_cntr):
    grad = np.gradient(f0_cntr)
    return stats.kurtosis(grad)


def get_Sskew(f0_cntr):
    return stats.skew(f0_cntr)


def get_Sdskew(f0_cntr):
    grad = np.gradient(f0_cntr)
    return stats.skew(grad)


def get_SVmeanRange(f0_cntr):
    in_voice = False
    cum_range = 0
    num_voice_seg = 0
    start_idx = -1
    for i in range(f0_cntr.shape[0]):
        if in_voice == False:
            if f0_cntr[i] != 0 and (f0_cntr[i: i + 5] != 0).all():
                in_voice = True
                num_voice_seg += 1
                start_idx = i
        elif in_voice == True:
            if f0_cntr[i] != 0:
                continue
            elif f0_cntr[i] == 0:
                in_voice = False
                cum_range += np.max(f0_cntr[start_idx:i]) - \
                    np.min(f0_cntr[start_idx:i])

    return cum_range / num_voice_seg


def get_SVmaxCurv(f0_cntr):

    in_voice = False
    max_curv = None
    num_voice_seg = 0
    start_idx = -1
    for i in range(f0_cntr.shape[0]):
        if in_voice == False:
            if f0_cntr[i] != 0 and (f0_cntr[i: i + 5] != 0).all():
                in_voice = True
                num_voice_seg += 1
                start_idx = i
        elif in_voice == True:
            if f0_cntr[i] != 0:
                continue
            elif f0_cntr[i] == 0:
                in_voice = False
                curv = np.polyfit(np.arange(i - start_idx),
                                  f0_cntr[start_idx:i], 2)[0]
                if max_curv == None or curv > max_curv:
                    max_curv = curv

    return curv


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
    "SQ75": get_SQ75,
    "Smedian": get_Smedian,
    "Sdmedian": get_Sdmedian,
    "Siqr": get_Siqr,
    "Sdiqr": get_Sdiqr,
    "Skurt": get_Skurt,
    "Sdkurt": get_Sdkurt,
    "Sskew": get_Sskew,
    "Sdskew": get_Sdskew,
    "SVmeanRange": get_SVmeanRange,
    "SVmaxCurv": get_SVmaxCurv
}
