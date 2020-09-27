
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation

from poliastro.core.propagation import farnocchia as farnocchia_fast

def farnocchia(k, r, v, tofs, **kwargs):
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
    # TODO: Rewrite to avoid iterating twice
    return (
        [result[0] for result in results] * u.km,
        [result[1] for result in results] * u.km / u.s,
    )

def propagate(orbit, time_of_flight, *, method=farnocchia, rtol=1e-10, **kwargs):
    """Propagate an orbit some time and return the result.
    Parameters
    ----------
    orbit : ~poliastro.twobody.Orbit
        Orbit object to propagate.
    time_of_flight : ~astropy.time.TimeDelta
        Time of propagation.
    method : callable, optional
        Propagation method, default to farnocchia.
    rtol : float, optional
        Relative tolerance, default to 1e-10.
    Returns
    -------
    astropy.coordinates.CartesianRepresentation
        Propagation coordinates.
    """

    rr, vv = method(
        orbit.attractor.k,
        orbit.r,
        orbit.v,
        time_of_flight.reshape(-1).to(u.s),
        rtol=rtol,
        **kwargs
    )

    # TODO: Turn these into unit tests
    assert rr.ndim == 2
    assert vv.ndim == 2

    cartesian = CartesianRepresentation(
        rr, differentials=CartesianDifferential(vv, xyz_axis=1), xyz_axis=1
    )

    return cartesian
