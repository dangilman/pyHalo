import numpy as np

__all__ = ['TabulatedMassFunction']


class TabulatedMassFunction(object):
    """
    Samples halos from a mass function specified by its values at a set of halo masses (nodes). Between nodes, dN/dm
    is a power law (it is interpolated linearly in log(m)-log(dN/dm)), so the mean number of halos, the first moment,
    the values returned by evaluate, and the masses returned by draw are exact and mutually consistent for this
    interpolant. A segment where dN/dm is zero at either node contains no halos.

    kwargs_model for from_redshift include:
    1) log_mlow: minimum halo mass to render (log base 10)
    2) log_mhigh: maximum halo mass to render (log base 10)
    3) log10m_table: log10 of the halo masses at which the mass function is tabulated [M_sun]
    4) z_table: the redshifts at which the mass function is tabulated (one redshift for a redshift-independent table)
    5) dndm_comoving_table: the comoving differential mass function dn/dm [Mpc^-3 M_sun^-1], with shape
    (len(log10m_table), len(z_table)); interpolated linearly in log(dn/dm) between redshifts
    6) LOS_normalization: rescales the amplitude of the mass function (default 1)
    7) m_pivot: the pivot mass for delta_power_law_index (default 10^8)
    8) delta_power_law_index: tilts the logarithmic slope of the mass function around m_pivot (default 0)
    9) draw_poisson: bool; draw the number of halos from a Poisson distribution
    """
    name = 'TABULATED'

    def __init__(self, log_mlow, log_mhigh, m_nodes, dN_dm_nodes, draw_poisson):
        """

        :param log_mlow: log10(minimum halo mass)
        :param log_mhigh: log10(maximum halo mass)
        :param m_nodes: increasing array of halo masses [M_sun] at which dN/dm is specified; must cover the range
        [10^log_mlow, 10^log_mhigh]
        :param dN_dm_nodes: the differential number of halos dN/dm at m_nodes [M_sun^-1], including the rendering volume
        :param draw_poisson: bool; whether to draw the number of halos from a Poisson distribution
        """
        m_nodes = np.asarray(m_nodes, dtype=float)
        dN_dm_nodes = np.asarray(dN_dm_nodes, dtype=float)
        if m_nodes.ndim != 1 or m_nodes.shape != dN_dm_nodes.shape or len(m_nodes) < 2:
            raise ValueError('m_nodes and dN_dm_nodes must be one-dimensional arrays of the same length (at least 2)')
        if np.any(np.diff(m_nodes) <= 0) or m_nodes[0] <= 0:
            raise ValueError('m_nodes must be positive and strictly increasing')
        if np.any(dN_dm_nodes < 0) or not np.all(np.isfinite(dN_dm_nodes)):
            raise ValueError('dN_dm_nodes must be finite and non-negative')
        self._logmlow = log_mlow
        self._logmhigh = log_mhigh
        self._draw_poisson = draw_poisson
        self._m_nodes_input = m_nodes
        self._dN_dm_nodes_input = dN_dm_nodes
        if log_mlow >= log_mhigh:
            self._m = np.array([])
            self._segment_counts = np.array([])
            self.n_mean = 0
            self.first_moment = 0
            return
        m_low, m_high = 10 ** log_mlow, 10 ** log_mhigh
        if m_low < m_nodes[0] * (1 - 1e-10) or m_high > m_nodes[-1] * (1 + 1e-10):
            raise ValueError('the mass function nodes, log10(m) in [%.3f, %.3f], do not cover the mass range '
                             '[%.3f, %.3f]' % (np.log10(m_nodes[0]), np.log10(m_nodes[-1]), log_mlow, log_mhigh))
        interior = (m_nodes > m_low) & (m_nodes < m_high)
        self._m = np.concatenate(([m_low], m_nodes[interior], [m_high]))
        self._dN_dm = np.concatenate((self._interpolate(np.array([m_low]), m_nodes, dN_dm_nodes),
                                      dN_dm_nodes[interior],
                                      self._interpolate(np.array([m_high]), m_nodes, dN_dm_nodes)))
        self._index, self._nonzero = self._segment_slopes(self._m, self._dN_dm)
        self._segment_counts = self._segment_moments(0)
        self.n_mean = float(np.sum(self._segment_counts))
        self.first_moment = float(np.sum(self._segment_moments(1)))

    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, kwargs_model):
        """
        Creates the mass function at the lens plane at redshift z from a table of the comoving mass function

        :param z: redshift of the lens plane
        :param delta_z: thickness of the lens plane
        :param geometry_class: an instance of Geometry
        :param kwargs_model: keyword arguments for the mass function (see the class docstring)
        :return: an instance of TabulatedMassFunction
        """
        log10m_table = np.asarray(kwargs_model['log10m_table'], dtype=float)
        z_table = np.atleast_1d(np.asarray(kwargs_model['z_table'], dtype=float))
        dndm_table = np.asarray(kwargs_model['dndm_comoving_table'], dtype=float)
        if dndm_table.ndim == 1:
            dndm_table = dndm_table[:, np.newaxis]
        if dndm_table.shape != (len(log10m_table), len(z_table)):
            raise ValueError('dndm_comoving_table must have shape (len(log10m_table), len(z_table))')
        if np.any(dndm_table <= 0):
            raise ValueError('dndm_comoving_table must be positive')
        if len(z_table) == 1:
            log_dndm = np.log(dndm_table[:, 0])
        else:
            if z < z_table[0] or z > z_table[-1]:
                raise ValueError('redshift %.3f outside the range of z_table [%.3f, %.3f]' %
                                 (z, z_table[0], z_table[-1]))
            j = min(int(np.searchsorted(z_table, z, side='right')) - 1, len(z_table) - 2)
            t = (z - z_table[j]) / (z_table[j + 1] - z_table[j])
            log_dndm = (1 - t) * np.log(dndm_table[:, j]) + t * np.log(dndm_table[:, j + 1])
        m = 10 ** log10m_table
        dN_dm = np.exp(log_dndm) * cls._volume_normalization(m, z, delta_z, geometry_class, kwargs_model)
        return TabulatedMassFunction(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], m, dN_dm,
                                     kwargs_model['draw_poisson'])

    @staticmethod
    def _volume_normalization(m, z, delta_z, geometry_class, kwargs_model):
        """
        The factor that converts the comoving mass function into the number of halos per unit mass in a lens plane:
        the comoving volume of the plane times LOS_normalization, with the tilt (m / m_pivot)^delta_power_law_index

        :param m: halo masses [M_sun]
        :param z: redshift of the lens plane
        :param delta_z: thickness of the lens plane
        :param geometry_class: an instance of Geometry
        :param kwargs_model: keyword arguments for the mass function
        :return: the factor at each mass
        """
        volume = geometry_class.volume_element_comoving(z, delta_z)
        normalization = kwargs_model.get('LOS_normalization', 1.0)
        delta_power_law_index = kwargs_model.get('delta_power_law_index', 0.0)
        m_pivot = kwargs_model.get('m_pivot', 10 ** 8)
        return volume * normalization * (m / m_pivot) ** delta_power_law_index

    def evaluate(self, m):
        """
        Evaluates the mass function; masses outside the nodes are extrapolated with the power law of the nearest segment
        :param m: halo mass [M_sun]
        :return: dN/dm [M_sun^-1]
        """
        scalar = np.ndim(m) == 0
        m = np.atleast_1d(np.asarray(m, dtype=float))
        if len(self._m) == 0:
            values = self._interpolate(m, self._m_nodes_input, self._dN_dm_nodes_input)
        else:
            values = self._interpolate(m, self._m, self._dN_dm)
        if scalar:
            return float(values[0])
        return values

    def draw(self):
        """
        Samples from the mass function
        :return: an array of halo masses [M_sun]
        """
        if self.n_mean == 0:
            return np.array([])
        if self._draw_poisson:
            ndraw = np.random.poisson(self.n_mean)
        else:
            ndraw = int(round(self.n_mean))
        if ndraw == 0:
            return np.array([])
        cdf = np.cumsum(self._segment_counts) / self.n_mean
        # segments that contain no halos have a flat cdf and are never selected
        segment = np.minimum(np.searchsorted(cdf, np.random.uniform(0, 1, ndraw), side='right'),
                             len(self._segment_counts) - 1)
        u = np.random.uniform(0, 1, ndraw)
        m_low = self._m[segment]
        log_ratio = np.log(self._m[segment + 1] / m_low)
        # within a segment dN/dm ~ m^index; invert the cumulative distribution analytically
        s = self._index[segment] + 1
        t = s * log_ratio
        linear = np.abs(t) < 1e-8
        log_m = np.empty(ndraw)
        log_m[linear] = u[linear] * log_ratio[linear]
        log_m[~linear] = np.log1p(u[~linear] * np.expm1(t[~linear])) / s[~linear]
        return m_low * np.exp(log_m)

    def _segment_moments(self, n):
        """
        :param n: the order of the moment
        :return: the integral of m^n dN/dm over each segment
        """
        m_low = self._m[:-1]
        log_ratio = np.log(self._m[1:] / m_low)
        t = (n + 1 + self._index) * log_ratio
        # integral of m^(n + index) from m_low to m_high, written as m_low^(n+1) * log_ratio * (e^t - 1) / t
        small = np.abs(t) < 1e-8
        factor = np.ones_like(t)
        factor[~small] = np.expm1(t[~small]) / t[~small]
        factor[small] = 1 + 0.5 * t[small]
        moments = self._dN_dm[:-1] * m_low ** (n + 1) * log_ratio * factor
        moments[~self._nonzero] = 0.0
        return moments

    @staticmethod
    def _segment_slopes(m, dN_dm):
        """
        :param m: node masses
        :param dN_dm: mass function at the nodes
        :return: the logarithmic slope of each segment, and a mask of segments with non-zero dN/dm at both ends
        """
        nonzero = (dN_dm[:-1] > 0) & (dN_dm[1:] > 0)
        index = np.zeros(len(m) - 1)
        index[nonzero] = np.log(dN_dm[1:][nonzero] / dN_dm[:-1][nonzero]) / np.log(m[1:][nonzero] / m[:-1][nonzero])
        return index, nonzero

    @classmethod
    def _interpolate(cls, m, m_nodes, dN_dm_nodes):
        """
        Interpolates the mass function between nodes with power laws, extrapolating with the end segments
        :param m: masses at which to evaluate the mass function
        :param m_nodes: node masses
        :param dN_dm_nodes: mass function at the nodes
        :return: dN/dm at m
        """
        index, nonzero = cls._segment_slopes(m_nodes, dN_dm_nodes)
        segment = np.clip(np.searchsorted(m_nodes, m, side='right') - 1, 0, len(m_nodes) - 2)
        values = np.zeros(len(m))
        keep = nonzero[segment]
        values[keep] = dN_dm_nodes[segment[keep]] * (m[keep] / m_nodes[segment[keep]]) ** index[segment[keep]]
        return values
