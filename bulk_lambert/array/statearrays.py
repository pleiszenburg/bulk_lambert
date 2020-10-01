from astropy import units as u

import numpy as np

from .._jit import jit
from ..elements import coe2mee, coe2rv, mee2coe, rv2coe


# from .elements import mean_motion, period
import numpy as np
def mean_motion(k, a):
    """Mean motion given body (k) and semimajor axis (a)."""
    return np.sqrt(k / abs(a ** 3)).to(1 / u.s) * u.rad
def period(k, a):
    """Period given body (k) and semimajor axis (a)."""
    n = mean_motion(k, a)
    return 2 * np.pi * u.rad / n


class BaseStateArray:
    """Base State class, meant to be subclassed."""

    def __init__(self, attractor, plane):
        """Constructor.
        Parameters
        ----------
        attractor : Body
            Main attractor.
        plane : ~poliastro.frames.enums.Planes
            Reference plane for the elements.
        """
        self._attractor = attractor
        self._plane = plane

    @property
    def plane(self):
        """Fundamental plane of the frame."""
        return self._plane

    @property
    def attractor(self):
        """Main attractor. """
        return self._attractor

    @property
    def n(self):
        """Mean motion. """
        return mean_motion(self.attractor.k, self.to_classical().a)

    @property
    def period(self):
        """Period of the orbit. """
        return period(self.attractor.k, self.to_classical().a)

    def to_vectors(self):
        """Converts to position and velocity vector representation.
        Returns
        -------
        RVState
        """
        raise NotImplementedError

    def to_classical(self):
        """Converts to classical orbital elements representation.
        Returns
        -------
        ClassicalState
        """
        raise NotImplementedError

    def to_equinoctial(self):
        """Converts to modified equinoctial elements representation.
        Returns
        -------
        ModifiedEquinoctialState
        """
        raise NotImplementedError


class ClassicalStateArray(BaseStateArray):
    """State defined by its classical orbital elements."""

    def __init__(self, attractor, p, ecc, inc, raan, argp, nu, plane):
        super().__init__(attractor, plane)

        assert p.shape == ecc.shape == inc.shape == raan.shape == argp.shape == nu.shape

        self._p = p
        self._ecc = ecc
        self._inc = inc
        self._raan = raan
        self._argp = argp
        self._nu = nu

    @property
    def p(self):
        """Semilatus rectum. """
        return self._p

    @property
    def a(self):
        """Semimajor axis. """
        return self.p / (1 - self.ecc ** 2)

    @property
    def ecc(self):
        """Eccentricity. """
        return self._ecc

    @property
    def inc(self):
        """Inclination. """
        return self._inc

    @property
    def raan(self):
        """Right ascension of the ascending node. """
        return self._raan

    @property
    def argp(self):
        """Argument of the perigee. """
        return self._argp

    @property
    def nu(self):
        """True anomaly. """
        return self._nu

    def to_vectors(self):
        """Converts to position and velocity vector representation."""

        r, v = np.zeros((self.p.shape[0], 3)), np.zeros((self.p.shape[0], 3))

        self._to_vectors(
            self.attractor.k.to(u.km ** 3 / u.s ** 2).value,
            self.p.to(u.km).value,
            self.ecc.value,
            self.inc.to(u.rad).value,
            self.raan.to(u.rad).value,
            self.argp.to(u.rad).value,
            self.nu.to(u.rad).value,
            r,
            v,
        )

        return RVStateArray(
            self.attractor,
            r * u.km,
            v * u.km / u.s,
            self.plane,
        )

    @staticmethod
    @jit
    def _to_vectors(
        k,
        p,
        ecc,
        inc,
        raan,
        argp,
        nu,
        r,
        v,
    ):

        for idx in range(p.shape[0]):

            r[idx, :], v[idx, :] = coe2rv(
                k,
                p[idx],
                ecc[idx],
                inc[idx],
                raan[idx],
                argp[idx],
                nu[idx],
            )

    def to_classical(self):
        """Converts to classical orbital elements representation."""
        return self

    def to_equinoctial(self):
        """Converts to modified equinoctial elements representation."""
        p, f, g, h, k, L = coe2mee(
            self.p.to(u.km).value,
            self.ecc.value,
            self.inc.to(u.rad).value,
            self.raan.to(u.rad).value,
            self.argp.to(u.rad).value,
            self.nu.to(u.rad).value,
        )

        return ModifiedEquinoctialStateArray(
            self.attractor,
            p * u.km,
            f * u.rad,
            g * u.rad,
            h * u.rad,
            k * u.rad,
            L * u.rad,
            self.plane,
        )


class RVStateArray(BaseStateArray):
    """State defined by its position and velocity vectors."""

    def __init__(self, attractor, r, v, plane):
        super().__init__(attractor, plane)

        assert r.shape == v.shape

        self._r = r
        self._v = v

    @property
    def r(self):
        """Position vector. """
        return self._r

    @property
    def v(self):
        """Velocity vector. """
        return self._v

    def to_vectors(self):
        """Converts to position and velocity vector representation."""
        return self

    def to_classical(self):
        """Converts to classical orbital elements representation."""

        p, ecc, inc, raan, argp, nu = (
            np.zeros((self._r.shape[0],)),
            np.zeros((self._r.shape[0],)),
            np.zeros((self._r.shape[0],)),
            np.zeros((self._r.shape[0],)),
            np.zeros((self._r.shape[0],)),
            np.zeros((self._r.shape[0],)),
        )

        self._to_classical(
            self.attractor.k.to(u.km ** 3 / u.s ** 2).value,
            self.r.to(u.km).value,
            self.v.to(u.km / u.s).value,
            p, ecc, inc, raan, argp, nu,
        )

        return ClassicalStateArray(
            self.attractor,
            p * u.km,
            ecc * u.one,
            inc * u.rad,
            raan * u.rad,
            argp * u.rad,
            nu * u.rad,
            self.plane,
        )

    @staticmethod
    @jit
    def _to_classical(
        k,
        r,
        v,
        p, ecc, inc, raan, argp, nu,
    ):

        for idx in range(r.shape[0]):
            p[idx], ecc[idx], inc[idx], raan[idx], argp[idx], nu[idx] = rv2coe(
                k,
                r[idx, :],
                v[idx, :],
            )



class ModifiedEquinoctialStateArray(BaseStateArray):
    def __init__(self, attractor, p, f, g, h, k, L, plane):
        super().__init__(attractor, plane)

        assert p.shape == f.shape == g.shape == h.shape == k.shape == L.shape

        self._p = p
        self._f = f
        self._g = g
        self._h = h
        self._k = k
        self._L = L

    @property
    def p(self):
        return self._p

    @property
    def f(self):
        return self._f

    @property
    def g(self):
        return self._g

    @property
    def h(self):
        return self._h

    @property
    def k(self):
        return self._k

    @property
    def L(self):
        return self._L

    def to_classical(self):

        p, ecc, inc, raan, argp, nu = (
            np.zeros((self.p.shape[0],)),
            np.zeros((self.p.shape[0],)),
            np.zeros((self.p.shape[0],)),
            np.zeros((self.p.shape[0],)),
            np.zeros((self.p.shape[0],)),
            np.zeros((self.p.shape[0],)),
        )

        self._to_classical(
            self.p.to(u.km).value,
            self.f.to(u.rad).value,
            self.g.to(u.rad).value,
            self.h.to(u.rad).value,
            self.k.to(u.rad).value,
            self.L.to(u.rad).value,
            p, ecc, inc, raan, argp, nu,
        )

        return ClassicalStateArray(
            self.attractor,
            p * u.km,
            ecc * u.one,
            inc * u.rad,
            raan * u.rad,
            argp * u.rad,
            nu * u.rad,
            self.plane,
        )

    @staticmethod
    @jit
    def _to_classical(
        p_,
        f,
        g,
        h,
        k,
        L,
        p, ecc, inc, raan, argp, nu,
    ):

        for idx in range(p_.shape[0]):
            p[idx], ecc[idx], inc[idx], raan[idx], argp[idx], nu[idx] = mee2coe(
                p_,
                f,
                g,
                h,
                k,
                L,
            )
