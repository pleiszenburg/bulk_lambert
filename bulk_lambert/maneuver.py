
import astropy.units as u

from numpy.linalg import norm

from .izzo import lambert_izzo

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
