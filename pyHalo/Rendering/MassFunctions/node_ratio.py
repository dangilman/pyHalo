"""
Halo mass functions equal to a reference mass function times a ratio R(m) = n / n_CDM specified at certain mass scales
(nodes, see pyHalo.node_ratios). The same ratio applies at every redshift.
"""
import numpy as np
from pyHalo.node_ratios import Log10RatioSpline, LOG10_M_NODES
from pyHalo.Rendering.MassFunctions.tabulated import TabulatedMassFunction
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen

__all__ = ['NodeRatioMassFunction', 'NodeRatioSubhaloMassFunction']


def _mass_grid(log_mlow, log_mhigh, nodes_per_dex):
    n_nodes = max(2, int(np.ceil((log_mhigh - log_mlow) * nodes_per_dex)) + 1)
    return np.logspace(log_mlow, log_mhigh, n_nodes)


class NodeRatioMassFunction(TabulatedMassFunction):
    """
    The field halo mass function at each lens plane: a CDM mass function times the ratio R(m), whose log10 is a
    cubic spline through its values at the mass nodes

    kwargs_model for this class include those of the CDM mass function class, and:
    1) log10_n_ratio_nodes: log10(n / n_CDM) at the nodes
    2) log10_m_nodes: log10 of the node masses (default 6, 8, 10)
    3) log10_n_ratio_floor: the smallest value of log10(n / n_CDM) (default -2); None for no floor
    4) cdm_mass_function_class: the CDM mass function class (default ShethTormen, as in the CDM preset model); it must
    have a from_redshift method and an evaluate method
    5) nodes_per_dex: number of nodes per decade in halo mass at which the mass function is tabulated (default 20)
    """
    name = 'NODE_RATIO'

    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, kwargs_model):
        """

        :param z: redshift of the lens plane
        :param delta_z: thickness of the lens plane
        :param geometry_class: an instance of Geometry
        :param kwargs_model: keyword arguments for the mass function (see the class docstring)
        :return: an instance of NodeRatioMassFunction
        """
        log_mlow, log_mhigh = kwargs_model['log_mlow'], kwargs_model['log_mhigh']
        draw_poisson = kwargs_model['draw_poisson']
        if log_mlow >= log_mhigh:
            return NodeRatioMassFunction(log_mlow, log_mhigh, np.array([1.0, 2.0]), np.zeros(2), draw_poisson)
        cdm_class = kwargs_model.get('cdm_mass_function_class', ShethTormen)
        mfunc_cdm = cdm_class.from_redshift(z, delta_z, geometry_class, kwargs_model)
        ratio = Log10RatioSpline(kwargs_model['log10_n_ratio_nodes'], kwargs_model.get('log10_m_nodes', LOG10_M_NODES),
                                 kwargs_model.get('log10_n_ratio_floor', -2.0))
        m = _mass_grid(log_mlow, log_mhigh, kwargs_model.get('nodes_per_dex', 20))
        dN_dm = mfunc_cdm.evaluate(m) * ratio.ratio(m)
        return NodeRatioMassFunction(log_mlow, log_mhigh, m, dN_dm, draw_poisson)


class NodeRatioSubhaloMassFunction(TabulatedMassFunction):
    """
    The subhalo mass function at infall: the power law of CDMPowerLaw times the ratio R(m) of NodeRatioMassFunction,

    dN/dm = normalization * m^power_law_index * R(m)

    For R(m) = 1, the mass function is the same as a CDM power law. The enhancement of the SHMF at infall is the same
    as the field halo mass function at each node.
    """
    name = 'NODE_RATIO_SHMF'

    def __init__(self, log_mlow, log_mhigh, power_law_index, draw_poisson, normalization, log10_n_ratio_nodes,
                 log10_m_nodes=LOG10_M_NODES, log10_n_ratio_floor=-2.0, nodes_per_dex=20, **kwargs):
        """

        :param log_mlow: log10(minimum halo mass)
        :param log_mhigh: log10(maximum halo mass)
        :param power_law_index: logarithmic slope of the CDM power law
        :param draw_poisson: bool; whether to draw the number of halos from a Poisson distribution
        :param normalization: the amplitude of the CDM power law, dN/dm = normalization * m^power_law_index
        :param log10_n_ratio_nodes: log10(n / n_CDM) at the nodes
        :param log10_m_nodes: log10 of the node masses
        :param log10_n_ratio_floor: the smallest value of log10(n / n_CDM); None for no floor
        :param nodes_per_dex: number of nodes per decade in halo mass at which the mass function is tabulated
        :param kwargs: other keyword arguments from the subhalo renderer, which are not used
        """
        if log_mlow >= log_mhigh:
            super(NodeRatioSubhaloMassFunction, self).__init__(log_mlow, log_mhigh, np.array([1.0, 2.0]),
                                                               np.zeros(2), draw_poisson)
            return
        ratio = Log10RatioSpline(log10_n_ratio_nodes, log10_m_nodes, log10_n_ratio_floor)
        m = _mass_grid(log_mlow, log_mhigh, nodes_per_dex)
        dN_dm = normalization * m ** power_law_index * ratio.ratio(m)
        super(NodeRatioSubhaloMassFunction, self).__init__(log_mlow, log_mhigh, m, dN_dm, draw_poisson)
