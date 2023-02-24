"""Simple power transformation classes."""
# pylint: disable=unused-variable

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END

# TODO this could be better organized...

EPS = 1e-8


###CTM @sync_numerical_libs
def yeojohnson(y, lam):
    """Yeo-Johnson tranform, batched in the first dimension."""
    y_in = y.astype(xp.float64)

    lam1 = xp.broadcast_to(lam, (y_in.shape[0], 1)).astype(xp.float64)

    ret = xp.empty(y.shape)
    zero_mask = xp.around(xp.ravel(lam1), 4) == 0.0
    two_mask = xp.around(xp.ravel(lam1), 4) == 2.0
    pos_mask = y_in >= 0.0
    zero_mask = xp.broadcast_to(zero_mask[:, None], pos_mask.shape)
    two_mask = xp.broadcast_to(two_mask[:, None], pos_mask.shape)
    lam1 = xp.broadcast_to(lam1, pos_mask.shape)

    ret[pos_mask] = ((y_in[pos_mask] + 1.0) ** lam1[pos_mask] - 1.0) / lam1[pos_mask]
    ret[pos_mask & zero_mask] = xp.log(y_in[pos_mask & zero_mask] + 1.0)

    ret[~pos_mask] = ((1.0 - y_in[~pos_mask]) ** (2.0 - lam1[~pos_mask]) - 1.0) / (lam1[~pos_mask] - 2.0)
    ret[(~pos_mask) & two_mask] = -xp.log(1.0 - y_in[(~pos_mask) & two_mask])

    return ret, lam1[:, 0][..., None]


###CTM @sync_numerical_libs
def inv_yeojohnson(y, lam):
    """Inverse Yeo-Johnson tranform, batched in the first dimension."""
    y_in = y.astype(xp.float64)

    lam1 = xp.broadcast_to(lam, (y_in.shape[0], 1)).astype(xp.float64)

    ret = xp.empty(y.shape)
    zero_mask = xp.around(xp.ravel(lam1), 4) == 0.0
    two_mask = xp.around(xp.ravel(lam1), 4) == 2.0
    pos_mask = y_in >= 0.0
    zero_mask = xp.broadcast_to(zero_mask[:, None], pos_mask.shape)
    two_mask = xp.broadcast_to(two_mask[:, None], pos_mask.shape)
    lam1 = xp.broadcast_to(lam1, pos_mask.shape)

    ret[pos_mask] = (lam1[pos_mask] * y_in[pos_mask] + 1.0) ** (1.0 / (lam1[pos_mask] + EPS)) - 1.0
    ret[pos_mask & zero_mask] = xp.exp(y_in[pos_mask & zero_mask]) - 1.0

    ret[~pos_mask] = -(((lam1[~pos_mask] - 2.0) * y_in[~pos_mask] + 1.0) ** (1.0 / (2.0 - lam1[~pos_mask]))) + 1.0
    ret[(~pos_mask) & two_mask] = -xp.exp(-y_in[(~pos_mask) & two_mask]) + 1.0

    return ret


###CTM @sync_numerical_libs
def boxcox(y, lam, lam2=None):
    """Box-Cox tranform, batched in the first dimension."""
    # TODO add axis param
    # if axis is None:
    #    a = xp.ravel(a)
    #    axis = 0
    axis = y.ndim - 1

    y_in = y.astype(xp.float64)

    lam1 = xp.broadcast_to(lam, (y_in.shape[0], 1)).astype(xp.float64)

    if lam2 is None:
        lam2 = 1.0 - xp.min(y_in, axis=axis, keepdims=True)

    ret = xp.empty(y.shape)
    zero_mask = xp.around(xp.ravel(lam1), 4) == 0.0

    ret[zero_mask] = xp.log(y_in[zero_mask] + lam2[zero_mask])

    ret[~zero_mask] = ((y_in[~zero_mask] + lam2[~zero_mask]) ** lam1[~zero_mask] - 1.0) / lam1[~zero_mask]

    return ret, lam1, lam2


###CTM @sync_numerical_libs
def inv_boxcox(y, lam1, lam2):
    """Inverse Box-Cox tranform, batched in the first dimension."""
    y_in = y.astype(xp.float64)

    ret = xp.empty(y.shape)
    zero_mask = xp.around(xp.ravel(lam1), 4) == 0.0
    ret[zero_mask] = xp.exp(y_in[zero_mask]) - lam2[zero_mask]

    ret[~zero_mask] = (lam1[~zero_mask] * y_in[~zero_mask] + 1.0) ** (1.0 / lam1[~zero_mask]) - lam2[~zero_mask]

    return ret


def norm_cdf(x, mu, sigma):
    """Normal distribution CDF, batched."""
    t = x - mu[:, None]
    y = 0.5 * xp.special.erfc(-t / (sigma[:, None] * xp.sqrt(2.0)))  # pylint: disable=no-member
    y[y > 1.0] = 1.0
    return y


###CTM @sync_numerical_libs
def fit_lam(y, yj=False, lam_range=(-2, 2, 0.1)):
    """Fit lambda of a power transform using grid search, taking the the most normally distributed result."""

    # TODO currently this just minimizes the KS-stat,
    # would might better to used shapiro-wilk or 'normaltest' but we'd need a batched version
    y_in = xp.atleast_2d(y)
    batch_size = y_in.shape[0]

    best_ks = xp.full(batch_size, xp.inf)
    best_ks_lam = xp.empty(batch_size)
    for lam in xp.around(xp.arange(*lam_range), 6):
        if yj:
            yp, _ = yeojohnson(y, lam)
        else:
            yp, _, _ = boxcox(y_in, lam)
        ys = xp.sort(yp, axis=1)
        cdf = xp.cumsum(ys, axis=1) / xp.sum(yp, axis=1, keepdims=True)
        ks = xp.max(xp.abs(cdf - norm_cdf(ys, xp.mean(yp, axis=1), xp.var(yp, axis=1))), axis=1)
        ks_mask = ks < best_ks
        best_ks[ks_mask] = ks[ks_mask]
        best_ks_lam[ks_mask] = lam
    return (best_ks, best_ks_lam)


class BoxCox:
    """Wrapper class for a Box-Cox transformer."""

    def __init__(
        self,
    ):
        """Init lambda storage."""
        self.lam1 = None
        self.lam2 = None

    def fit(self, y):
        """Fit the batched 1d variables in y, store the lambdas for the inv transform."""
        ks = fit_lam(y, yj=False)
        ret, self.lam1, self.lam2 = boxcox(y, ks[1][:, None])
        return ret

    def inv(self, y):
        """Inverse tranform using the fitted lambda values."""
        return inv_boxcox(y, self.lam1, self.lam2)


class YeoJohnson:
    """Wrapper class for a Yeo-Johnson transformer."""

    def __init__(
        self,
    ):
        """Init lambda storage."""
        self.lam1 = None

    def fit(self, y):
        """Fit the batched 1d variables in y, store the lambdas for the inv transform."""
        ks = fit_lam(y, yj=True)
        ret, self.lam1 = yeojohnson(y, ks[1][:, None])
        return ret

    def inv(self, y):
        """Inverse tranform using the fitted lambda values."""
        return inv_yeojohnson(y, self.lam1)
