# -*- coding: utf-8 -*-

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import astropy.units as u
from astropy.time import TimeDelta

import numpy as np
from numpy.linalg import norm

from .iod import izzo as izzo_fast
from ._jit import jit
from .farnocchia import farnocchia as farnocchia_fast
from .util import time_range

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PROPAGATE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def propagate(
    orbit,
    epochs,
):
    """Propagate an orbit some epochs and return the results.
    Parameters
    ----------
    orbit : ~poliastro.twobody.Orbit
        Orbit object to propagate.
    epochs : ~astropy.time.Time
        Time of propagation.
    Returns
    -------
    rr[idx, dim], vv[idx, dim]
    """

    times_of_flight = epochs - orbit.epoch

    # STRIP UNITS
    k = orbit.attractor.k.to(u.km ** 3 / u.s ** 2).value
    r0 = orbit.r.to(u.km).value
    v0 = orbit.v.to(u.km / u.s).value
    tofs = times_of_flight.to(u.s).value

    # Allocate memory
    shape = (times_of_flight.shape[0], 3)
    rr = np.zeros(shape, dtype = 'f8')
    vv = np.zeros(shape, dtype = 'f8')

    # Compute
    for idx, tof in enumerate(tofs):
        rr[idx, :], vv[idx, :] = farnocchia_fast(k, r0, v0, tof)

    # ADD UNITS
    return rr * u.km, vv * (u.km / u.s)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LAMBERT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def lambert(
    attractor,

    rr_i,
    vv_i,
    rr_f,
    vv_f,

    epoch_start,
    epoch_stop,

    tof_steps_start: int,
    tof_steps_stop: int,

    M_max = 2,
    numiter = 35,
    rtol = 1e-8,
):
    """Computes Lambert maneuver between two different points.
    Parameters
    ----------
    attractor: Body

    rr_i: Initial location
    vv_i: Initial velocity
    rr_f: Final location
    vv_f: Final velocity

    epoch_start: ~astropy.time.Time
        Time for rr and vv.
    epoch_stop: ~astropy.time.Time
        Time for rr and vv.

    tof_steps_start : int
    tof_steps_stop : int

    M : int, optional
        Number of full revolutions, default to 0.
    numiter : int, optional
        Maximum number of iterations, default to 35.
    rtol : float, optional
        Relative tolerance of the algorithm, default to 1e-8.
    """

    assert rr_i.shape == rr_f.shape == vv_i.shape == vv_f.shape
    assert 0 < tof_steps_start
    assert tof_steps_start <= tof_steps_stop

    # times
    epochs = time_range(
        start = epoch_start,
        end = epoch_stop,
    )
    dt = epochs[1] - epochs[0]
    epochs_select = epochs[:-tof_steps_stop]
    epochs_select_len = epochs_select.shape[0]
    tofs = TimeDelta(
        [(dt * idx).value for idx in range(tof_steps_start, tof_steps_stop + 1)],
        scale = dt.scale,
        format = dt.format,
    )

    # STRIP UNITS
    k_ = attractor.k.to(u.km ** 3 / u.s ** 2).value
    rr_i_ = rr_i.to(u.km).value
    vv_i_ = vv_i.to(u.km / u.s).value
    rr_f_ = rr_f.to(u.km).value
    vv_f_ = vv_f.to(u.km / u.s).value
    tofs_ = tofs.to(u.s).value

    # Allocate memory
    shape = (
        epochs_select_len,
        len(range(tof_steps_start, tof_steps_stop + 1)),
        3,
    )
    dv1 = np.zeros(shape, dtype = 'f8')
    dv2 = np.zeros(shape, dtype = 'f8')
    dv = np.zeros(shape[:-1], dtype = 'f8')
    MM = np.zeros(shape[:-1], dtype = 'u8')

    # Compute
    for epoch_idx in range(epochs_select_len):

        for tof_idx, (tof_step, tof_) in enumerate(zip(range(tof_steps_start, tof_steps_stop + 1), tofs_)):

            solutions = []

            for M in range(0, M_max + 1):

                solutions.extend(izzo_fast_(
                    k_,
                    rr_i_[epoch_idx, :],
                    rr_f_[epoch_idx + tof_step, :],
                    tof_,
                    M,
                    numiter,
                    rtol,
                ))

            solutions = [
                (
                    M,
                    v1 - vv_i_[epoch_idx, :],
                    vv_f_[epoch_idx + tof_step, :] - v2,
                )
                for M, v1, v2 in solutions
            ]
            solutions = [
                (
                    M,
                    v1,
                    v2,
                    norm(v1) + norm(v2),
                )
                for M, v1, v2 in solutions
            ]
            best = min(solutions, key = lambda solution: solutions[3])
            MM[epoch_idx, tof_idx], dv1[epoch_idx, tof_idx, :], dv2[epoch_idx, tof_idx, :], dv[epoch_idx, tof_idx] = best

    # ADD UNITS
    return epochs_select, tofs, dv * (u.km / u.s), MM, dv1 * (u.km / u.s), dv2 * (u.km / u.s)

@jit
def izzo_fast_(k, r1, r2, tof, M, numiter, rtol):

    for v1, v2 in izzo_fast(k, r1, r2, tof, M, numiter, rtol):
        yield M, v1, v2
