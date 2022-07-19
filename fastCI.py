import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

def credibility_interval(samples, weights=None, level=0.68, method="hpd"):
    """Compute the credibility interval of weighted samples. Based on
    linear interpolation of the cumulative density function, thus expected
    discretization errors on the scale of distances between samples.

    Parameters
    ----------
    samples: array
        Samples to compute the credibility interval of.
    samples: array, optional
        Weights corresponding to samples. Default: ones
    level: float, optional
        Credibility level (probability, <1). Default: 0.68
    method: str, optional
        Which definition of interval to use. Default: 'hpd'
        * hpd: Calculate highest posterior density (HPD) interval,
          equivalent to iso-pdf interval (interval with same probability
          density at each end) if distribution is unimodal.
        * ll, ul: Lower limit, upper limit. One-sided limits for which
          `level` fraction of the samples lie above / below the limit.
        * et: Equal-tailed interval with the same fraction of samples
          below and above the interval region.

    Returns
    -------
    limit: tuple or float
        Tuple [lower, upper] for hpd/et, or float for ll/ul
    """
    if level>=1:
        raise ValueError('level must be <1, got {0:.2f}'.format(level))
    if len(np.shape(samples)) != 1:
        raise ValueError('Support only 1D arrays for samples')
    if weights is not None and np.shape(samples) != np.shape(weights):
        raise ValueError('Shape of samples and weights differs')

    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = np.array(samples)[order]
    weights = np.array(weights)[order]/np.sum(weights)
    # Compute inverse cumulative distribution function
    cumsum = np.cumsum(weights)
    S = np.array([np.min(samples), *samples, np.max(samples)])
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
    invCDF = interp1d(CDF, S)

    if method == "hpd":
        # Find smallest interval
        distance = lambda Y, level=level: invCDF(Y+level)-invCDF(Y)
        res = minimize_scalar(distance, bounds=(0, 1-level), method="Bounded")
        return np.array([invCDF(res.x), invCDF(res.x+level)])
    elif method == "ll":
        # Get value from which we reach the desired level
        return invCDF(1-level)
    elif method == "ul":
        # Get value to which we reach the desired level
        return invCDF(level)
    elif method == "et":
        return np.array([invCDF((1-level)/2), invCDF((1+level)/2)])
    else:
        raise ValueError("Method '{0:}' unknown".format(method))