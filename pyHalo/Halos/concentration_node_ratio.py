"""
A concentration-mass relation equal to a CDM relation times a ratio c / c_CDM specified at mass nodes (see
pyHalo.node_ratios). The same ratio applies at every redshift.
"""
import numpy
from pyHalo.Halos.concentration import _ConcentrationTurnover
from pyHalo.node_ratios import Log10RatioSpline, LOG10_M_NODES

__all__ = ['ConcentrationNodeRatio']


class ConcentrationNodeRatio(_ConcentrationTurnover):
    """
    c(M, z) = c_CDM(M, z) * 10^S(log10 M), where S is a natural cubic spline through log10(c / c_CDM) at the mass nodes,
    held at the end values outside the nodes. The scatter is drawn for the CDM concentration (as in the other models
    built on a CDM relation), so the ratio rescales the whole distribution of concentrations at fixed mass; the
    universal minimum concentration applies after the ratio.
    """
    name = 'NODE_RATIO'

    def __init__(self, cosmo, concentration_cdm_class, log10_c_ratio_nodes, log10_m_nodes=LOG10_M_NODES,
                 scatter=True, scatter_dex=0.15, kwargs_cdm=None):
        """

        :param cosmo: an instance of astropy cosmology
        :param concentration_cdm_class: a concentration class for CDM (for example ConcentrationLudlow)
        :param log10_c_ratio_nodes: log10(c / c_CDM) at the nodes
        :param log10_m_nodes: log10 of the node masses [M_sun]
        :param scatter: bool; whether to include scatter in the concentration-mass relation
        :param scatter_dex: scatter in concentration in dex
        :param kwargs_cdm: keyword arguments for the CDM concentration class
        """
        kwargs_cdm = {} if kwargs_cdm is None else dict(kwargs_cdm)
        kwargs_cdm.setdefault('scatter', scatter)
        kwargs_cdm.setdefault('scatter_dex', scatter_dex)
        cdm_concentration = concentration_cdm_class(cosmo, **kwargs_cdm)
        self._log10_ratio = Log10RatioSpline(log10_c_ratio_nodes, log10_m_nodes)
        super(ConcentrationNodeRatio, self).__init__(cdm_concentration)

    def suppression(self, m, z):
        """
        The ratio c / c_CDM (named suppression for consistency with the other models built on a CDM relation)

        :param m: halo mass [M_sun]; float or array
        :param z: halo redshift (the ratio does not depend on redshift)
        :return: c / c_CDM
        """
        ratio = self._log10_ratio.ratio(m)
        if numpy.ndim(m) == 0:
            return float(ratio)
        return ratio
