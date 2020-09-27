import numpy as np
from numpy import cross, pi
from numpy.linalg import norm

import astropy.units as u

from poliastro.core._jit import jit


@jit
def hyp2f1b(x):
    """Hypergeometric function 2F1(3, 1, 5/2, x), see [Battin].
    .. todo::
        Add more information about this function
    Note
    ----
    More information about hypergeometric function can be checked at
    https://en.wikipedia.org/wiki/Hypergeometric_function
    """
    if x >= 1.0:
        return np.inf
    else:
        res = 1.0
        term = 1.0
        ii = 0
        while True:
            term = term * (3 + ii) * (1 + ii) / (5 / 2 + ii) * x / (ii + 1)
            res_old = res
            res += term
            if res_old == res:
                return res
            ii += 1

@jit
def izzo_fast(k, r1, r2, tof, M, numiter, rtol): # izzo
    """ Aplies izzo algorithm to solve Lambert's problem.
    Parameters
    ----------
    k: float
        Gravitational Constant
    r1: ~numpy.array
        Initial position vector
    r2: ~numpy.array
        Final position vector
    tof: float
        Time of flight between both positions
    M: int
        Number of revolutions
    numiter: int
        Numbert of iterations
    rtol: float
        Error tolerance
    Returns
    -------
    v1: ~numpy.array
        Initial velocity vector
    v2: ~numpy.array
        Final velocity vector
    """

    # Check preconditions
    assert tof > 0
    assert k > 0

    # Check collinearity of r1 and r2
    if np.all(cross(r1, r2) == 0):
        raise ValueError("Lambert solution cannot be computed for collinear vectors")

    # Chord
    c = r2 - r1
    c_norm, r1_norm, r2_norm = norm(c), norm(r1), norm(r2)

    # Semiperimeter
    s = (r1_norm + r2_norm + c_norm) * 0.5

    # Versors
    i_r1, i_r2 = r1 / r1_norm, r2 / r2_norm
    i_h = cross(i_r1, i_r2)
    i_h = i_h / norm(i_h)  # Fixed from paper

    # Geometry of the problem
    ll = np.sqrt(1 - min(1.0, c_norm / s))

    if i_h[2] < 0:
        ll = -ll
        i_h = -i_h

    i_t1, i_t2 = cross(i_h, i_r1), cross(i_h, i_r2)  # Fixed from paper

    # Non dimensional time of flight
    T = np.sqrt(2 * k / s ** 3) * tof

    # Find solutions
    xy = _find_xy(ll, T, M, numiter, rtol)

    # Reconstruct
    gamma = np.sqrt(k * s / 2)
    rho = (r1_norm - r2_norm) / c_norm
    sigma = np.sqrt(1 - rho ** 2)

    for x, y in xy:
        V_r1, V_r2, V_t1, V_t2 = _reconstruct(
            x, y, r1_norm, r2_norm, ll, gamma, rho, sigma
        )
        v1 = V_r1 * i_r1 + V_t1 * i_t1
        v2 = V_r2 * i_r2 + V_t2 * i_t2
        yield v1, v2

@jit
def _reconstruct(x, y, r1, r2, ll, gamma, rho, sigma):
    """Reconstruct solution velocity vectors.
    """
    V_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1
    V_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2
    V_t1 = gamma * sigma * (y + ll * x) / r1
    V_t2 = gamma * sigma * (y + ll * x) / r2
    return [V_r1, V_r2, V_t1, V_t2]

@jit
def _find_xy(ll, T, M, numiter, rtol):
    """Computes all x, y for given number of revolutions.
    """
    # For abs(ll) == 1 the derivative is not continuous
    assert abs(ll) < 1
    assert T > 0  # Mistake on original paper

    M_max = np.floor(T / pi)
    T_00 = np.arccos(ll) + ll * np.sqrt(1 - ll ** 2)  # T_xM

    # Refine maximum number of revolutions if necessary
    if T < T_00 + M_max * pi and M_max > 0:
        _, T_min = _compute_T_min(ll, M_max, numiter, rtol)
        if T < T_min:
            M_max -= 1

    # Check if a feasible solution exist for the given number of revolutions
    # This departs from the original paper in that we do not compute all solutions
    if M > M_max:
        raise ValueError("No feasible solution, try lower M")

    # Initial guess
    for x_0 in _initial_guess(T, ll, M):
        # Start Householder iterations from x_0 and find x, y
        x = _householder(x_0, T, ll, M, rtol, numiter)
        y = _compute_y(x, ll)

        yield x, y

@jit
def _compute_y(x, ll):
    """Computes y.
    """
    return np.sqrt(1 - ll ** 2 * (1 - x ** 2))

@jit
def _compute_psi(x, y, ll):
    """Computes psi.
    "The auxiliary angle psi is computed using Eq.(17) by the appropriate
    inverse function"
    """
    if -1 <= x < 1:
        # Elliptic motion
        # Use arc cosine to avoid numerical errors
        return np.arccos(x * y + ll * (1 - x ** 2))
    elif x > 1:
        # Hyperbolic motion
        # The hyperbolic sine is bijective
        return np.arcsinh((y - x * ll) * np.sqrt(x ** 2 - 1))
    else:
        # Parabolic motion
        return 0.0

@jit
def _tof_equation(x, T0, ll, M):
    """Time of flight equation.
    """
    return _tof_equation_y(x, _compute_y(x, ll), T0, ll, M)

@jit
def _tof_equation_y(x, y, T0, ll, M):
    """Time of flight equation with externally computated y.
    """
    if M == 0 and np.sqrt(0.6) < x < np.sqrt(1.4):
        eta = y - ll * x
        S_1 = (1 - ll - x * eta) * 0.5
        Q = 4 / 3 * hyp2f1b(S_1)
        T_ = (eta ** 3 * Q + 4 * ll * eta) * 0.5
    else:
        psi = _compute_psi(x, y, ll)
        T_ = np.divide(
            np.divide(psi + M * pi, np.sqrt(np.abs(1 - x ** 2))) - x + ll * y,
            (1 - x ** 2),
        )

    return T_ - T0

@jit
def _tof_equation_p(x, y, T, ll):
    # TODO: What about derivatives when x approaches 1?
    return (3 * T * x - 2 + 2 * ll ** 3 * x / y) / (1 - x ** 2)

@jit
def _tof_equation_p2(x, y, T, dT, ll):
    return (3 * T + 5 * x * dT + 2 * (1 - ll ** 2) * ll ** 3 / y ** 3) / (1 - x ** 2)

@jit
def _tof_equation_p3(x, y, _, dT, ddT, ll):
    return (7 * x * ddT + 8 * dT - 6 * (1 - ll ** 2) * ll ** 5 * x / y ** 5) / (
        1 - x ** 2
    )

@jit
def _compute_T_min(ll, M, numiter, rtol):
    """Compute minimum T.
    """
    if ll == 1:
        x_T_min = 0.0
        T_min = _tof_equation(x_T_min, 0.0, ll, M)
    else:
        if M == 0:
            x_T_min = np.inf
            T_min = 0.0
        else:
            # Set x_i > 0 to avoid problems at ll = -1
            x_i = 0.1
            T_i = _tof_equation(x_i, 0.0, ll, M)
            x_T_min = _halley(x_i, T_i, ll, rtol, numiter)
            T_min = _tof_equation(x_T_min, 0.0, ll, M)

    return [x_T_min, T_min]

@jit
def _initial_guess(T, ll, M):
    """Initial guess.
    """
    if M == 0:
        # Single revolution
        T_0 = np.arccos(ll) + ll * np.sqrt(1 - ll ** 2) + M * pi  # Equation 19
        T_1 = 2 * (1 - ll ** 3) / 3  # Equation 21
        if T >= T_0:
            x_0 = (T_0 / T) ** (2 / 3) - 1
        elif T < T_1:
            x_0 = 5 / 2 * T_1 / T * (T_1 - T) / (1 - ll ** 5) + 1
        else:
            # This is the real condition, which is not exactly equivalent
            # elif T_1 < T < T_0
            x_0 = (T_0 / T) ** (np.log2(T_1 / T_0)) - 1

        return [x_0]
    else:
        # Multiple revolution
        x_0l = (((M * pi + pi) / (8 * T)) ** (2 / 3) - 1) / (
            ((M * pi + pi) / (8 * T)) ** (2 / 3) + 1
        )
        x_0r = (((8 * T) / (M * pi)) ** (2 / 3) - 1) / (
            ((8 * T) / (M * pi)) ** (2 / 3) + 1
        )

        return [x_0l, x_0r]

@jit
def _halley(p0, T0, ll, tol, maxiter):
    """Find a minimum of time of flight equation using the Halley method.
    Note
    ----
    This function is private because it assumes a calling convention specific to
    this module and is not really reusable.
    """
    for ii in range(maxiter):
        y = _compute_y(p0, ll)
        fder = _tof_equation_p(p0, y, T0, ll)
        fder2 = _tof_equation_p2(p0, y, T0, fder, ll)
        if fder2 == 0:
            raise RuntimeError("Derivative was zero")
        fder3 = _tof_equation_p3(p0, y, T0, fder, fder2, ll)

        # Halley step (cubic)
        p = p0 - 2 * fder * fder2 / (2 * fder2 ** 2 - fder * fder3)

        if abs(p - p0) < tol:
            return p
        p0 = p

    raise RuntimeError("Failed to converge")

@jit
def _householder(p0, T0, ll, M, tol, maxiter):
    """Find a zero of time of flight equation using the Householder method.
    Note
    ----
    This function is private because it assumes a calling convention specific to
    this module and is not really reusable.
    """
    for ii in range(maxiter):
        y = _compute_y(p0, ll)
        fval = _tof_equation_y(p0, y, T0, ll, M)
        T = fval + T0
        fder = _tof_equation_p(p0, y, T, ll)
        fder2 = _tof_equation_p2(p0, y, T, fder, ll)
        fder3 = _tof_equation_p3(p0, y, T, fder, fder2, ll)

        # Householder step (quartic)
        p = p0 - fval * (
            (fder ** 2 - fval * fder2 / 2)
            / (fder * (fder ** 2 - fval * fder2) + fder3 * fval ** 2 / 6)
        )

        if abs(p - p0) < tol:
            return p
        p0 = p

    raise RuntimeError("Failed to converge")

kms = u.km / u.s

def lambert_izzo(k, r0, r, tof, M=0, numiter=35, rtol=1e-8): # lambert
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

class Maneuver:
    r"""Class to represent a Maneuver.
    Each ``Maneuver`` consists on a list of impulses :math:`\Delta v_i`
    (changes in velocity) each one applied at a certain instant :math:`t_i`.
    You can access them directly indexing the ``Maneuver`` object itself.
    >>> man = Maneuver((0 * u.s, [1, 0, 0] * u.km / u.s),
    ... (10 * u.s, [1, 0, 0] * u.km / u.s))
    >>> man[0]
    (<Quantity 0. s>, <Quantity [1., 0., 0.] km / s>)
    >>> man.impulses[1]
    (<Quantity 10. s>, <Quantity [1., 0., 0.] km / s>)
    """

    def __init__(self, *args):
        r"""Constructor.
        Parameters
        ----------
        impulses : list
            List of pairs (delta_time, delta_velocity)
        """

        self.impulses = args
        # HACK: Change API or validation code
        _dts, _dvs = zip(*args)
        self._dts, self._dvs = self._initialize(
            [(_dt * u.one).value for _dt in _dts] * (_dts[0] * u.one).unit,
            [(_dv * u.one).value for _dv in _dvs] * (_dvs[0] * u.one).unit,
        )
        try:
            if not all(len(dv) == 3 for dv in self._dvs):
                raise TypeError
        except TypeError:
            raise ValueError("Delta-V must be three dimensions vectors")

    def __repr__(self):
        return f"Number of impulses: {len(self.impulses)}, Total cost: {self.get_total_cost():.6f}"

    @u.quantity_input(dts=u.s, dvs=u.m / u.s)
    def _initialize(self, dts, dvs):
        return dts, dvs

    # def __getitem__(self, key):
    #     return self.impulses[key]

    @classmethod
    def lambert(cls, orbit_i, orbit_f, method=lambert_izzo, short=True, **kwargs):
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
        sols = list(method(k, r_i, r_f, tof, **kwargs))

        # Return short or long solution
        if short:
            dv_a, dv_b = sols[0]
        else:
            dv_a, dv_b = sols[-1]

        return cls(
            (0 * u.s, (dv_a - orbit_i.v).decompose()),
            (tof.to(u.s), (orbit_f.v - dv_b).decompose()),
        )

    def get_total_time(self):
        """Returns total time of the maneuver.
        """
        total_time = sum(self._dts, 0 * u.s)
        return total_time

    def get_total_cost(self):
        """Returns total cost of the maneuver.
        """
        dvs = [norm(dv) for dv in self._dvs]
        return sum(dvs, 0 * u.km / u.s)
