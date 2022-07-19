# fastCI
Compute credible intervals quickly.

[Jump to code](https://github.com/Stefan-Heimersheim/fastCI/blob/main/README.md#full-code)

## What is a credible interval?
A [credible interval](https://en.wikipedia.org/wiki/Credible_interval) is an interval containing a certain fraction (e.g. 68%) of the probability volume of a distribution. Or, in other words, the true value of an unknown parameter lies within the _68% credible interval_ with 68% probability.

Technically any interval is _a_ credible interval, but typically we are interested in certain sizes and types of credible interval. Physicists and Astronomers frequently use [68%, 95%, and 99.7% intervals](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule) (colloquially referred to as 1,2, and 3 sigma intervals), inspired by the normal distribution (though the precise values, e.g. 68.27%, differ slightly). We also often call them "confidence intervals" and quote values of parameters at "confidence levels", but usually this refers to credible intervals. Jake VanderPlas is giving a good explanation of the differences [here](http://jakevdp.github.io/blog/2014/06/12/frequentism-and-bayesianism-3-confidence-credibility/).

In terms of types of intervals, there are 3 types of credible interval I see often, matching [Wikipedia's list](https://en.wikipedia.org/wiki/Credible_interval). I will use 68% credible intervals for illustration:
* Highest posterior density (HPD) or iso-probability-density (iso-pdf) intervals (mostly equivalent): The smallest (i.e. densest) interval containing 68% of the probability volume. This is also the interval where the probability density is equal at the interval boundaries, thus it is sometimes referred to as "waterline method" -- imagine a waterline where the credibility interval is wherever the probability density function lies above that line:
![illustration](https://github.com/Stefan-Heimersheim/fastCI/blob/main/illustrations/highest-probability-density-interval.png?raw=true)
* Equal-tailed interval: The interval containing 68% of the probability volume with the same probability volume below and above the interval, i.e. 16% on each side.
![illustration](https://github.com/Stefan-Heimersheim/fastCI/blob/main/illustrations/equal-tailed-interval.png?raw=true)
* Mean-centered interval: The interval centered around the mean, i.e. an interval [mean-a, mean+a] containing 68% of the probability volume. This is often quotes as "mean +/- error (68% confidence)". Although note that this notation also can mean "mean +/- error" not necessarily containing 68% probability.
![illustration](https://github.com/Stefan-Heimersheim/fastCI/blob/main/illustrations/mean-centered-interval.png?raw=true)
(Note: The interval happens to look similar to the previous one, this is a coincidence and not generally true.)

## Computing the iso-pdf interval efficiently
Common methods rely on interpolating the probability density function (PDF) via kernel density estimation (KDE) to derive the credible interval. This (a) introduces a dependency on the KDE used, and (b) is often quite slow, especially for large sample sizes.

Instead, we can use the samples themselves to derive this interval, realizing that iso-pdf and HPD are independent, and HPD does not require an interpolated PDF. Here is an example of a Normal distribution, along with a histogram of 1000 samples and the 68% HPD interval derived from these samples:
![example histogram](https://github.com/Stefan-Heimersheim/fastCI/blob/main/illustrations/high_sample_example.png?raw=true)

To understand how, let us look at this smaller 10-sample demonstration. Starting with common sense, we can already see what the smallest intervals containing 60 and 70% of the data are, i.e. these are the respective HPD intervals. We also notice possible uncertainties, coming from (a) the precise number (68%) not matching the availble fractions (green arrow, 10 points -> 50%, or 60%, or 70% etc.), and from (b) the concrete realization of the data (red arrows, the position of those data points was somewhat random).
![low-sample example](https://github.com/Stefan-Heimersheim/fastCI/blob/main/illustrations/low_sample_example.png?raw=true)

Also, this feels a bit _unfair_, including points exactly at the edges of the intervals. And with _unfair_ I mean it's relying on the low sample size and probably underestimating the interval size. We could also choose these intervals differently (looking at 70% only now):
![low-sample example-v2](https://github.com/Stefan-Heimersheim/fastCI/blob/main/illustrations/low_sample_example_v2.png?raw=true)
The green line seems like a fair middle ground, extending the one side but not the other -- and choosing the steeper side to extend to aim for the highest posterior denity.

Now to the way to (quickly) calculate this. Key is that, for any start point of an interval `a`, we can easily find the end point `b` that makes `[a,b]` the, say, 68% confidence interval. The cumulative density function (CDF) gives us the probability volume, so we want to fulfill `CDF(b) - CDF(a) = 68%`. Since we have discrete samples we can easily calculate the CDF (`np.cumsum`) and invert it to get the inverse CDF.
![CDF](https://github.com/Stefan-Heimersheim/fastCI/blob/main/illustrations/CDF_illustration.png?raw=true)
Note that there are two ways to compute this, from 0 or from 1, but the lines are equivalent since we only look at differences between two points of the CDF, not the absolute value.

Now we can get the interval boundary `b` corresponding to `a` containing 68% of the probability volume as `b = invCDF(CDF(a) + 0.68)`. In practice we can optimize over `Y = CDF(a)` such that `b - a = invCDF(Y+0.68) - invCDF(Y)` is mimized.

Thinking graphically, we are scanning different y-levels `Y` of the plot below and check how large the x-distance between the x-value corresponding to `Y` and `Y+0.68` is.
![Graphical explanation](https://github.com/Stefan-Heimersheim/fastCI/blob/main/illustrations/CDF_distances.png?raw=true)

Going back to our common-sense consideration beforehand, the automated solution (using CDF interpolation) seems quite close to our green line ("fair" guess for 70%), so this seems to make sense. And we know that this is the minimal interval to include 68% of the interpolated CDF.
![low-sample example-again](https://github.com/Stefan-Heimersheim/fastCI/blob/main/illustrations/low_sample_again.png?raw=true)

## Full code
The function in `fastCI.py` can be used to compute the HPD and equal-tailed intervals, as well as lower and upper limits (percentiles). It also includes some checks to make sure all arguments are OK.
```python
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
```

## Simple version
A couple of quick-to-read lines, just to compute the HPD interval:
```python
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

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
    invCDF = interp1d(CDF, S)
    # Find smallest interval
    distance = lambda Y, level=level: invCDF(Y+level)-invCDF(Y)
    res = minimize_scalar(distance, bounds=(0, 1-level), method="Bounded")
    return np.array([invCDF(res.x), invCDF(res.x+level)])
```
