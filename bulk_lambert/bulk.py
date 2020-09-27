
import astropy.units as u
kms = u.km / u.s

from .iod import izzo_fast
from .farnocchia import farnocchia as farnocchia_fast

def _farnocchia_strip(k, r, v, tofs):
    """Propagates orbit.
    Parameters
    ----------
    k : ~astropy.units.Quantity
        Standard gravitational parameter of the attractor.
    r : ~astropy.units.Quantity
        Position vector.
    v : ~astropy.units.Quantity
        Velocity vector.
    tofs : ~astropy.units.Quantity
        Array of times to propagate.
    Returns
    -------
    rr : ~astropy.units.Quantity
        Propagated position vectors.
    vv : ~astropy.units.Quantity
        Propagated velocity vectors.
    """
    k = k.to(u.km ** 3 / u.s ** 2).value
    r0 = r.to(u.km).value
    v0 = v.to(u.km / u.s).value
    tofs = tofs.to(u.s).value

    results = [farnocchia_fast(k, r0, v0, tof) for tof in tofs]

    return (
        [result[0] for result in results] * u.km,
        [result[1] for result in results] * u.km / u.s,
    ) # TODO: Rewrite to avoid iterating twice

def propagate(orbit, time_of_flight):
    """Propagate an orbit some time and return the result.
    Parameters
    ----------
    orbit : ~poliastro.twobody.Orbit
        Orbit object to propagate.
    time_of_flight : ~astropy.time.TimeDelta
        Time of propagation.
    rtol : float, optional
        Relative tolerance, default to 1e-10.
    Returns
    -------
    astropy.coordinates.CartesianRepresentation
        Propagation coordinates.
    """

    rr, vv = _farnocchia_strip(
        orbit.attractor.k,
        orbit.r,
        orbit.v,
        time_of_flight.reshape(-1).to(u.s), # TODO
    )

    # TODO: Turn these into unit tests
    assert rr.ndim == 2
    assert vv.ndim == 2

    cartesian = CartesianRepresentation(
        rr, differentials=CartesianDifferential(vv, xyz_axis=1), xyz_axis=1
    )

    return cartesian

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
