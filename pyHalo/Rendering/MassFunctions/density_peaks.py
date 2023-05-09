from copy import deepcopy
import numpy as np
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw, WDMPowerLaw, MixedWDMPowerLaw
from colossus.lss.mass_function import massFunction

__all__ = ['ShethTormen', 'ShethTormenMixedWDM', 'ShethTormenTurnover']


class ShethTormen(CDMPowerLaw):
    """
    This class samples from the Sheth-Tormen halo mass function

    kwargs_model for this class include:
    1) log_mlow: minimum halo mass to render (log base 10)
    2) log_mhigh: maximum halo mass to render (log base 10)
    3) m_pivot: the pivot mass; the logarithmic slope is defined around the pivot
    4) LOS_normalization: rescales the ammplitude of the mass function at the pivot scale
    5) delta_power_law_index: adjusts the logarithmic slope of the mass function around the pivot scale
    """
    name = 'SHETH_TORMEN'
    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, kwargs_model):
        """

        :param z:
        :param delta_z:
        :param geometry_class:
        :param kwargs_model:
        :return:
        """
        _ = geometry_class.cosmo.colossus
        m_pivot = kwargs_model['m_pivot']
        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 10)
        m_h = m * h
        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / m_pivot), np.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + kwargs_model['delta_power_law_index']
        norm_dv = 10 ** coeffs[1] / (m_pivot**plaw_index)
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = kwargs_model['LOS_normalization'] * norm_dv * volume_element_comoving
        return ShethTormen(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                           kwargs_model['draw_poisson'], normalization)

class ShethTormenTurnover(WDMPowerLaw):
    """
    This class samples from the Sheth-Tormen halo mass function with a break at a scale 10^log_mc

    kwargs_model for this class include:
    1) log_mlow: minimum halo mass to render (log base 10)
    2) log_mhigh: maximum halo mass to render (log base 10)
    3) m_pivot: the pivot mass; the logarithmic slope is defined around the pivot
    4) LOS_normalization: rescales the ammplitude of the mass function at the pivot scale
    5) delta_power_law_index: adjusts the logarithmic slope of the mass function around the pivot scale
    6) log_mc: log 10 of the break scale
    7) a_wdm: shifts the position of the cutoff (see expression below)
    8) b_wdm: the inner exponent of the cutoff, determines the sharpness of the cut (see expression below)
    9) c_wdm: the outer exponent of the cutoff, determines the logarithmic slope below log_mc together with b_wdm

    cutoff has the functional form (1 + a_wdm * (m_c/m) ^ b_wdm) ^ c_wdm
    where m_c = 10^log_mc
    """
    name = 'SHETH_TORMEN'
    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, kwargs_model):
        """

        :param z:
        :param delta_z:
        :param geometry_class:
        :param kwargs_model:
        :return:
        """
        _ = geometry_class.cosmo.colossus
        m_pivot = kwargs_model['m_pivot']
        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 10)
        m_h = m * h
        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / m_pivot), np.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + kwargs_model['delta_power_law_index']
        norm_dv = 10 ** coeffs[1] / (m_pivot ** plaw_index)
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = kwargs_model['LOS_normalization'] * norm_dv * volume_element_comoving
        return ShethTormenTurnover(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                           kwargs_model['draw_poisson'], normalization, kwargs_model['log_mc'],
                                   kwargs_model['a_wdm'], kwargs_model['b_wdm'], kwargs_model['c_wdm'])

class ShethTormenMixedWDM(MixedWDMPowerLaw):
    """
    This class samples from the Sheth-Tormen halo mass function with a break at a scale 10^log_mc for mixed warm and
    cold dark matter

    kwargs_model for this class include:
    1) log_mlow: minimum halo mass to render (log base 10)
    2) log_mhigh: maximum halo mass to render (log base 10)
    3) m_pivot: the pivot mass; the logarithmic slope is defined around the pivot
    Note that this only has affect when delta_power_law_index != 0
    4) log_mc: log 10 of the break scale
    5) a_wdm: shifts the position of the cutoff (see expression below)
    6) b_wdm: the inner exponent of the cutoff, determines the sharpness of the cut (see expression below)
    7) c_wdm: the outer exponent of the cutoff, determines the logarithmic slope below log_mc together with b_wdm

    cutoff has the functional form (1 + a_wdm * (m_c/m) ^ b_wdm) ^ c_wdm
    where m_c = 10^log_mc
    """
    name = 'SHETH_TORMEN'
    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, kwargs_model):
        """

        :param z:
        :param delta_z:
        :param geometry_class:
        :param kwargs_model:
        :return:
        """
        colossus_cosmo = geometry_class.cosmo.colossus
        m_pivot = kwargs_model['m_pivot']
        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 10)
        m_h = m * h
        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / m_pivot), np.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + kwargs_model['delta_power_law_index']
        norm_dv = 10 ** coeffs[1]
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = kwargs_model['LOS_normalization'] * norm_dv * volume_element_comoving / (m_pivot ** plaw_index)
        return ShethTormenMixedWDM(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                                         kwargs_model['draw_poisson'], normalization, kwargs_model['log_mc'],
                                   kwargs_model['a_wdm'], kwargs_model['b_wdm'],
                                   kwargs_model['c_wdm'], kwargs_model['mixed_DM_frac'])

