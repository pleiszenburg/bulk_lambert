# -*- coding: utf-8 -*-

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import astropy.units as u
kms = u.km / u.s

import numpy as np

from .iod import izzo as izzo_fast
from .farnocchia import farnocchia as farnocchia_fast

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PROPAGATE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def propagate(
    orbit,
    times_of_flight,
):
    """Propagate an orbit some times and return the results.
    Parameters
    ----------
    orbit : ~poliastro.twobody.Orbit
        Orbit object to propagate.
    times_of_flight : ~astropy.time.TimeDelta
        Time of propagation (time deltas).
    Returns
    -------
    rr, vv
    """

    k = orbit.attractor.k.to(u.km ** 3 / u.s ** 2).value
    r0 = orbit.r.to(u.km).value
    v0 = orbit.v.to(u.km / u.s).value
    tofs = times_of_flight.to(u.s).value

    rr = np.zeros((times_of_flight.shape[0], 3), dtype = 'f8')
    vv = np.zeros((times_of_flight.shape[0], 3), dtype = 'f8')

    for idx, tof in enumerate(tofs):
        rr[idx, :], vv[idx, :] = farnocchia_fast(k, r0, v0, tof)

    return rr * u.km, vv * u.km / u.s

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LAMBERT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _izzo_strip(k, r0, r, tof, M=0, numiter=35, rtol=1e-8): # lambert
    """Solves the Lambert problem using the Izzo algorithm.
    .. versionadded:: 0.5.0
    Parameters
    ----------
    k : ~astropy.units.Quantity
        Gravitational constant of main attractor (km^3 / s^2).
    r0 : ~astropy.units.Quantity
        Initial position (km).
    r : ~astropy.units.Quantity
        Final position (km).
    tof : ~astropy.units.Quantity
        Time of flight (s).
    M : int, optional
        Number of full revolutions, default to 0.
    numiter : int, optional
        Maximum number of iterations, default to 35.
    rtol : float, optional
        Relative tolerance of the algorithm, default to 1e-8.
    Yields
    ------
    v0, v : tuple
        Pair of velocity solutions.
    """
    k_ = k.to(u.km ** 3 / u.s ** 2).value
    r0_ = r0.to(u.km).value
    r_ = r.to(u.km).value
    tof_ = tof.to(u.s).value

    sols = izzo_fast(k_, r0_, r_, tof_, M, numiter, rtol)

    for v0, v in sols:
        yield v0 << kms, v << kms

def lambert(
    orbit_i,
    orbit_f,
    epochs,
    tofs,
    short=True,
    **kwargs,
):
    """Computes Lambert maneuver between two different points.
    Parameters
    ----------
    orbit_i: ~poliastro.twobody.Orbit
        Initial orbit
    orbit_f: ~poliastro.twobody.Orbit
        Final orbit
    method: function
        Method for solving Lambert's problem
    short: keyword, boolean
        Selects between short and long solution
    """

    # Get initial algorithm conditions
    k = orbit_i.attractor.k
    r_i = orbit_i.r
    r_f = orbit_f.r

    # Time of flight is solved by subtracting both orbit epochs
    tof = orbit_f.epoch - orbit_i.epoch

    # Compute all possible solutions to the Lambert transfer
    sols = list(_izzo_strip(k, r_i, r_f, tof, **kwargs))

    # Return short or long solution
    if short:
        dv_a, dv_b = sols[0]
    else:
        dv_a, dv_b = sols[-1]

    return (
        (0 * u.s, (dv_a - orbit_i.v).decompose()), # solution #1, decomposed unit
        (tof.to(u.s), (orbit_f.v - dv_b).decompose()), # solution #2, decomposed unit
    )
