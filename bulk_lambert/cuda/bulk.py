# -*- coding: utf-8 -*-

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from numba import cuda

import astropy.units as u

import numpy as np

from ..elements import rv2coe # no CUDA
from ..farnocchia import delta_t_from_nu # no CUDA

from .elements import coe2rv # CUDA device function
from .farnocchia import nu_from_delta_t # CUDA device function

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
    _farnocchia_wrapper(k, r0, v0, tofs, rr, vv)

    # ADD UNITS
    return rr * u.km, vv * (u.km / u.s)


def _farnocchia_wrapper(
    k, # scalar
    r0, v0, # vectors
    tofs, # vector
    rr, vv, # arrays of vectors
):

    # get the initial true anomaly and orbit parameters that are constant over time
    p, ecc, inc, raan, argp, nu0 = rv2coe(k, r0, v0)
    q = p / (1 + ecc)

    delta_t0 = delta_t_from_nu(nu0, ecc, k, q)

    _farnocchia_kernel(
        tofs, # vector
        delta_t0, # scalar
        k, p, ecc, inc, raan, argp, q, # scalars
        rr, vv, # arrays of vectors
    )


@cuda.jit(device=False)
def _farnocchia_kernel(
    tofs, # vector
    delta_t0, # scalar
    k, p, ecc, inc, raan, argp, q, # scalars
    rr, vv, # arrays of vectors
):

    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    idx = tx + ty * bw

    # Early exit if we're out of bounds
    if tofs.shape[0] >= idx:
        return

    delta_t = delta_t0 + tofs[idx]

    nu = nu_from_delta_t(delta_t, ecc, k, q)

    coe2rv(
        k, p, ecc, inc, raan, argp, nu,
        _r, _r_buffer1, _r_buffer2, # 3x3
        _pqw, _ijk, # 2x3
        )
    rr[idx, :], vv[idx, :] = _ijk[0, :], _ijk[1, :]