"""Utility to interpolate/extrapolate values between different sets of binned age groups."""

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END
from .extrapolate import interp_extrap


###CTM @sync_numerical_libs
def age_bin_interp(age_bins_new, age_bins, y):
    """Interpolate parameters define in age groups to a new set of age groups."""
    # TODO we should probably account for population for the 65+ type bins...
    x_bins_new = xp.array(age_bins_new)
    x_bins = xp.array(age_bins)
    y = xp.array(y)
    if (x_bins_new.shape != x_bins.shape) or xp.any(x_bins_new != x_bins):
        x_mean_new = xp.mean(x_bins_new, axis=1)
        x_mean = xp.mean(x_bins, axis=1)
        return interp_extrap(x_mean_new, x_mean, y)
    return y
