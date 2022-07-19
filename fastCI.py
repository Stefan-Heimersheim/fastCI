import numpy as np
import scipy.interpolate as sip
import scipy.optimize as sop

def credibility_interval(samples, weights=None, level=0.68):
    assert level<1, "Level >= 1!"
    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = np.array(samples)[order]
    weights = np.array(weights)[order]/np.sum(weights)
    # Compute inverse cumulative distribution function
    cumsum = np.cumsum(weights)
    S = np.array([np.min(samples), *samples, np.max(samples)])
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
    invCDF = sip.interp1d(CDF, S)
    # Find smallest interval
    distance = lambda Y, level=level: invCDF(Y+level)-invCDF(Y)
    res = sop.minimize_scalar(distance, bounds=(0, 1-level), method="Bounded")
    return np.array([invCDF(res.x), invCDF(res.x+level)])