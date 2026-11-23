"""
Ratios of a parametric reltion, such as the halo mass function or the concentration-mass relation, to a reference model.
The ratios are specified at a few halo masses scales (nodes) and interpolated between them using a cubic spline.
"""
import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ['LOG10_M_NODES', 'Log10RatioSpline']

# log10 of the default node masses [M_sun]
LOG10_M_NODES = (6.0, 8.0, 10.0)


class Log10RatioSpline(object):
    """
    log10 of a ratio to CDM as a function of log10(M)
    """

    def __init__(self, log10_ratio_nodes, log10_m_nodes=LOG10_M_NODES, log10_floor=None):
        """

        :param log10_ratio_nodes: log10 of the ratio at each node
        :param log10_m_nodes: log10 of the node masses [M_sun]; at least two, strictly increasing
        :param log10_floor: the smallest value the log10 of the ratio can take; None for no floor
        """
        x = np.asarray(log10_m_nodes, dtype=float)
        y = np.asarray(log10_ratio_nodes, dtype=float)
        if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
            raise ValueError('log10_ratio_nodes and log10_m_nodes must be one-dimensional and of the same length '
                             '(at least two nodes)')
        if np.any(np.diff(x) <= 0):
            raise ValueError('log10_m_nodes must be strictly increasing')
        if not np.all(np.isfinite(y)):
            raise ValueError('log10_ratio_nodes must be finite')
        self.log10_m_nodes = x
        self.log10_ratio_nodes = y
        self.log10_floor = log10_floor
        self._spline = CubicSpline(x, y, bc_type='natural')

    def __call__(self, log10m):
        """

        :param log10m: log10 of the halo mass [M_sun]; float or array
        :return: log10 of the ratio, with the shape of log10m
        """
        x = np.clip(log10m, self.log10_m_nodes[0], self.log10_m_nodes[-1])
        y = self._spline(x)
        if self.log10_floor is not None:
            y = np.maximum(y, self.log10_floor)
        if np.ndim(log10m) == 0:
            return float(y)
        return y

    def ratio(self, m):
        """

        :param m: halo mass [M_sun]; float or array
        :return: the ratio to CDM, with the shape of m
        """
        return 10 ** self(np.log10(m))
