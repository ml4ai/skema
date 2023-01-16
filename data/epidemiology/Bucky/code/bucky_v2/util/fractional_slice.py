"""Utility to fractionally slice an xp array, keeping the fractional part between the int indices."""
###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END


###CTM @sync_numerical_libs
def frac_last_n_vals(arr, n, axis=0, offset=0):  # TODO assumes come from end of array currently
    """Return the last n values of an array; if n is a float, including fractional amounts."""
    int_slice_ind = tuple(
        [slice(None)] * (axis)
        + [slice(-int(n + offset), -int(xp.ceil(offset)) or None)]
        + [slice(None)] * (arr.ndim - axis - 1),
    )
    ret = arr[int_slice_ind]
    # handle fractional element before the standard slice
    if (n + offset) % 1:
        frac_slice_ind = tuple(
            [slice(None)] * (axis)
            + [slice(-int(n + offset + 1), -int(n + offset))]
            + [slice(None)] * (arr.ndim - axis - 1),
        )
        ret = xp.concatenate((((n + offset) % 1) * arr[frac_slice_ind], ret), axis=axis)
    # handle fractional element after the standard slice
    if offset % 1:
        frac_slice_ind = tuple(
            [slice(None)] * (axis)
            + [slice(-int(offset + 1), -int(offset) or None)]
            + [slice(None)] * (arr.ndim - axis - 1),
        )
        ret = xp.concatenate((ret, (1.0 - (offset % 1)) * arr[frac_slice_ind]), axis=axis)

    return ret
