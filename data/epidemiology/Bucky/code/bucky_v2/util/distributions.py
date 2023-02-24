"""Provide probability distributions used by the model that aren't in numpy/cupy."""

import numpy as np
import scipy.special as sc

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END


def kumaraswamy_invcdf(a, b, u):
    """Inverse CDF of the Kumaraswamy distribution."""
    return (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)


def approx_betaincinv(alp1, alp2, u):
    """Approximate betaincinv using Kumaraswamy after converting the params s.t. means and modes are equal."""
    a = alp1
    b = ((alp1 - 1.0) ** (1.0 - alp1) * (alp1 + alp2 - 2.0) ** alp1 + 1) / alp1
    return kumaraswamy_invcdf(a, xp.real(b), u)


###CTM @sync_numerical_libs
def approx_mPERT(mu, a=0.0, b=1.0, gamma=4.0):
    """Approximate sample from an mPERT distribution that uses a Kumaraswamy distrib in place of the incbeta.

    Notes
    -----
    Supports Cupy.
    """
    mu, a, b = xp.atleast_1d(mu, a, b)
    alp1 = 1.0 + gamma * ((mu - a) / (b - a))
    alp2 = 1.0 + gamma * ((b - mu) / (b - a))
    u = xp.random.random_sample(mu.shape)
    alp3 = approx_betaincinv(alp1.astype(xp.float64), alp2.astype(xp.float64), u)
    return (b - a) * alp3 + a


# TODO only works on cpu atm
# we'd need to implement betaincinv ourselves in cupy
def mPERT(mu, a=0.0, b=1.0, gamma=4.0, var=None):
    """Provide a vectorized Modified PERT distribution.

    Parameters
    ----------
    mu : float or ndarray
        Mean value for the PERT distribution.
    a : float or ndarray
        Lower bound for the distribution.
    b : float or ndarray
        Upper bound for the distribution.
    gamma : float or ndarray
        Shape paramter.
    var : float, ndarray or None
        Variance of the distribution. If var != None,
        gamma will be calcuated to meet the desired variance.

    Returns
    -------
    out : float or ndarray
        Samples drawn from the specified mPERT distribution.
        Shape is the broadcasted shape of the the input parameters.

    """
    mu, a, b = np.atleast_1d(mu, a, b)
    if var is not None:
        gamma = (mu - a) * (b - mu) / var - 3.0
    alp1 = 1.0 + gamma * ((mu - a) / (b - a))
    alp2 = 1.0 + gamma * ((b - mu) / (b - a))
    u = np.random.random_sample(mu.shape)
    alp3 = sc.betaincinv(alp1, alp2, u)  # pylint: disable=no-member
    return (b - a) * alp3 + a


###CTM @sync_numerical_libs
def truncnorm(loc=0.0, scale=1.0, size=None, a_min=None, a_max=None):
    """Provide a vectorized truncnorm implementation that is compatible with cupy.

    The output is calculated by using the numpy/cupy random.normal() and
    truncted via rejection sampling. The interface is intended to mirror
    the scipy implementation of truncnorm.

    Parameters
    ----------
    loc:
    scale:
    size:
    a_min:
    a_max:

    Returns
    -------
    ndarray:
    """

    ret = xp.random.normal(loc, scale, size)
    ret = xp.atleast_1d(ret)
    if a_min is None:
        a_min = xp.array(-xp.inf)
    if a_max is None:
        a_max = xp.array(xp.inf)

    while True:
        valid = (ret > a_min) & (ret < a_max)
        if xp.atleast_1d(valid).all():
            return ret
        ret[~valid] = xp.atleast_1d(xp.random.normal(loc, scale, size))[~valid]


def truncnorm_from_CI(CI, size=1, a_min=None, a_max=None):
    """Truncnorm implementation that first derives mean and standard deviation from a 95% confidence interval."""
    lower, upper = CI
    std95 = xp.sqrt(1.0 / 0.05)
    mean = (upper + lower) / 2.0
    stddev = (upper - lower) / std95 / 2.0
    return truncnorm(mean, stddev, size, a_min, a_max)
