"""Provides are version of xp.interp that will perform polynomial extrapolation for all OOB x values."""

###CTM from joblib import Memory

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END

# from ..util.read_config import bucky_cfg

# memory = Memory(bucky_cfg["cache_dir"], verbose=0)


# @memory.cache
###CTM @sync_numerical_libs
def _cached_polyfit(extra, *args, **kwargs): ### TF: 'extra' argument added to handle generation 
    """Joblib memory cached version of xp.polyfit()"""
    return xp.polyfit(*args, **kwargs)


###CTM @sync_numerical_libs
def interp_extrap(x, x1, yp, order=2, n_pts=None):
    """Interp function with polynomial extrapolation."""
    # pylint: disable=invalid-unary-operand-type
    if n_pts is None:
        n_pts = order + 1
        if len(yp) < n_pts:
            raise ValueError(f"Not enough data points ({len(yp)}) for requested polyfit size ({n_pts})")
    y = xp.interp(x, x1, yp)
    if xp.any(x > x1[-1]):
        p = xp.poly1d(_cached_polyfit(x1[-n_pts:], yp[-n_pts:], order))
        y[x > x1[-1]] = p(x[x > x1[-1]])
    if xp.any(x < x1[0]):
        p = xp.poly1d(_cached_polyfit(x1[:n_pts], yp[:n_pts], order))
        y[x < x1[0]] = p(x[x < x1[0]])
    return y
