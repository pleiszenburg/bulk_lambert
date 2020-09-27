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
    times,
):
    """Propagate an orbit some times and return the results.
    Parameters
    ----------
    orbit : ~poliastro.twobody.Orbit
        Orbit object to propagate.
    times_of_flight : ~astropy.time.Time
        Time of propagation.
    Returns
    -------
    rr[idx, dim], vv[idx, dim]
    """

    times_of_flight = times - orbit.epoch

    # STRIP UNITS
    k = orbit.attractor.k.to(u.km ** 3 / u.s ** 2).value
    r0 = orbit.r.to(u.km).value
    v0 = orbit.v.to(u.km / u.s).value
    tofs = times_of_flight.to(u.s).value

    # Allocate memory
    rr = np.zeros((times_of_flight.shape[0], 3), dtype = 'f8')
    vv = np.zeros((times_of_flight.shape[0], 3), dtype = 'f8')

    # Compute
    for idx, tof in enumerate(tofs):
        rr[idx, :], vv[idx, :] = farnocchia_fast(k, r0, v0, tof)

    # ADD UNITS
    return rr * u.km, vv * u.km / u.s

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LAMBERT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def lambert(
    orbit_i,
    orbit_f,
    epochs,
    tofs,
    short=True,
    M=0,
    numiter=35,
    rtol=1e-8,
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

    M : int, optional
        Number of full revolutions, default to 0.
    numiter : int, optional
        Maximum number of iterations, default to 35.
    rtol : float, optional
        Relative tolerance of the algorithm, default to 1e-8.
    """

    # Get initial algorithm conditions
    k = orbit_i.attractor.k
    r_i = orbit_i.r
    r_f = orbit_f.r

    # Time of flight is solved by subtracting both orbit epochs
    tof = orbit_f.epoch - orbit_i.epoch

    # STRIP UNITS
    k_ = k.to(u.km ** 3 / u.s ** 2).value
    r_i_ = r_i.to(u.km).value # r0_
    r_f_ = r.to(u.km).value # r_
    tof_ = tof.to(u.s).value

    # Compute
    sols = list(izzo_fast(k_, r_i_, r_f_, tof_, M, numiter, rtol))

    # ADD UNITS
    sols = [((v0 << kms), (v << kms)) for v0, v in sols]

    # Return short or long solution
    if short:
        dv_a, dv_b = sols[0]
    else:
        dv_a, dv_b = sols[-1]

    return (
        (0 * u.s, (dv_a - orbit_i.v).decompose()), # solution #1, decomposed unit
        (tof.to(u.s), (orbit_f.v - dv_b).decompose()), # solution #2, decomposed unit
    )
