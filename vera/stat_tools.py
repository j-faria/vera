import numpy as np


def wmean(a, e):
    """ Weighted mean of array `a`, with uncertainty given by `e`.
        
    Parameters
    ----------
    a : array
        Array containing data
    e : array
        Uncertainties on `a`.
        The weighted rms is calculated using the weighted mean, where the 
        weights are equal to 1/e**2
    """
    return np.average(a, weights=1/e**2)


def rms(a):
    """ Root mean square of array `a`"""
    return np.sqrt((a**2).mean())


def wrms(a, e):
    """ 
    Weighted root mean square of array `a`, with uncertanty given by `e` 
    
    Parameters
    ----------
    a : array
        Array containing data
    e : array
        Uncertainties on `a`.
        The weighted rms is calculated using the weighted mean, where the 
        weights are equal to 1/e**2
    """
    w = 1/e**2
    return np.sqrt(np.sum(w*(a - np.average(a, weights=w))**2) / sum(w))


def multi_average(a, obs, axis=None, weights=None):
    av = []
    for obsi in np.unique(obs):
        mask = obs == obsi
        if weights is None:
            av.append(np.average(a[mask], axis=axis))
        else:
            av.append(np.average(a[mask], axis=axis, weights=weights[mask]))
    return np.array(av)


def false_alarm_level_gatspy(system, *args, **kwargs):
    from astropy.timeseries import LombScargle
    s = system
    if not s.GLS['gatspy']:
        raise ValueError('periodogram was not calculated with gatspy')
    
    t = s.GLS['model'].t.copy()
    y = s.GLS['model'].y.copy()
    dy = s.GLS['model'].dy.copy()

    y -= s.GLS['model'].ymean_
    return LombScargle(t, y, dy).false_alarm_level(*args, **kwargs)
