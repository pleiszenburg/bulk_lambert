# -*- coding: utf-8 -*-

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import astropy.units as u
from astropy.time import Time # TimeDelta

import numpy as np
# from numpy.linalg import norm

# from .iod import izzo as izzo_fast
from ._jit import jit
from .elements import coe2rv, rv2coe
from .farnocchia import (
    delta_t_from_nu,
    nu_from_delta_t,
)
# from .util import time_range

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PROPAGATE
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def propagate_many(
    orbits,
    epoch,
):
    """Propagate many orbits ...
    Parameters
    ----------
    orbits : list of poliastro.twobody.Orbit
        Orbit objects to propagate.
    epoch : ~astropy.time.Time
        Time of propagation.
    Returns
    -------
    rr[idx, dim], vv[idx, dim]
    """

    orbit_epochs = Time([orbit.epoch for orbit in orbits])
    times_of_flight = epoch - orbit_epochs

    # STRIP UNITS
    ks = np.array([orbit.attractor.k.to(u.km ** 3 / u.s ** 2).value for orbit in orbits], dtype = 'f8')
    r0s = np.array([orbit.r.to(u.km).value for orbit in orbits], dtype = 'f8')
    v0s = np.array([orbit.v.to(u.km / u.s).value for orbit in orbits], dtype = 'f8')
    tofs = times_of_flight.to(u.s).value

    # Allocate memory
    shape = (len(orbits), 3)
    rr = np.zeros(shape, dtype = 'f8')
    vv = np.zeros(shape, dtype = 'f8')

    # Compute
    _farnocchia_many(ks, r0s, v0s, tofs, rr, vv)

    # ADD UNITS
    return rr * u.km, vv * (u.km / u.s)

@jit
def _farnocchia_many(ks, r0s, v0s, tofs, rr, vv):

    for k, r0, v0, tof, r, v in zip(ks, r0s, v0s, tofs, rr, vv):

        # get the initial true anomaly and orbit parameters that are constant over time
        p, ecc, inc, raan, argp, nu0 = rv2coe(k, r0, v0)
        q = p / (1 + ecc)

        delta_t0 = delta_t_from_nu(nu0, ecc, k, q)
        delta_t = delta_t0 + tof

        nu = nu_from_delta_t(delta_t, ecc, k, q)

        r[:], v[:] = coe2rv(k, p, ecc, inc, raan, argp, nu)
