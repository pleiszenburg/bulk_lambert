
import numpy as np
from numpy import cos, sin

from astropy.time import Time

from ._jit import jit


@jit
def rotation_matrix(angle, axis):
    c = cos(angle)
    s = sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    elif axis == 2:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Invalid axis: must be one of 'x', 'y' or 'z'")


def time_range(start, *, periods=50, spacing=None, end=None, format=None, scale=None):
    """Generates range of astronomical times.
    .. versionadded:: 0.8.0
    Parameters
    ----------
    periods : int, optional
        Number of periods, default to 50.
    spacing : Time or Quantity, optional
        Spacing between periods, optional.
    end : Time or equivalent, optional
        End date.
    Returns
    -------
    Time
        Array of time values.
    """
    start = Time(start, format=format, scale=scale)

    if spacing is not None and end is None:
        result = start + spacing * np.arange(0, periods)

    elif end is not None and spacing is None:
        end = Time(end, format=format, scale=scale)
        result = start + (end - start) * np.linspace(0, 1, periods)

    else:
        raise ValueError("Either 'end' or 'spacing' must be specified")

    return result
