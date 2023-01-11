"""Provide some generic utility functions that operate on numpy/cupy arrays"""
###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END


###CTM @sync_numerical_libs
def rolling_window(a, window_size, center=False, axis=0, pad=True, pad_mode="reflect", reflect_type="even", freq=1):
    """Use stride_tricks to add an extra dim on the end of an ndarray for each elements window."""

    if pad:
        pad_before = xp.zeros(len(a.shape), dtype=xp.int32)
        pad_after = xp.zeros(len(a.shape), dtype=xp.int32)
        if center:
            # only allow odd sized centered windows
            if not (window_size % 2):
                raise ValueError
            pad_size = window_size // 2
            pad_before[axis] = pad_size
            pad_after[axis] = pad_size
        else:
            pad_before[axis] = window_size - 1

        padding = list(zip(list(xp.to_cpu(pad_before)), list(xp.to_cpu(pad_after))))
        a = xp.pad(a, padding, mode=pad_mode, reflect_type=reflect_type)

    shape = list(a.shape)
    shape[axis] = a.shape[axis] - window_size + 1
    shape = tuple(shape) + (window_size,)
    strides = a.strides + (a.strides[axis],)
    freq_inds = [slice(None)] * (axis) + [slice(0, shape[axis], freq)] + [slice(None)] * (a.ndim - axis - 1)
    return xp.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[tuple(freq_inds)]


def unbroadcast(array):
    """Undo a broadcasted array view, squashing the broadcasted dims.

    Given an array, return a new array that is the smallest subset of the
    original array that can be re-broadcasted back to the original array.

    See https://stackoverflow.com/questions/40845769/un-broadcasting-numpy-arrays
    for more details.
    """

    if array.ndim == 0:
        return array

    array = array[tuple((slice(0, 1) if stride == 0 else slice(None)) for stride in array.strides)]

    # Remove leading ones, which are not needed in numpy broadcasting.
    first_not_unity = next((i for (i, s) in enumerate(array.shape) if s > 1), array.ndim)

    return array.reshape(array.shape[first_not_unity:])
