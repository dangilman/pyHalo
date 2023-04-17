from copy import deepcopy
import numpy as np
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw, WDMPowerLaw, MixedWDMPowerLaw
from colossus.lss.mass_function import massFunction


class ShethTormen(CDMPowerLaw):
    """
    This class generates masses from a delta function normalized with respect to a
    background density, a mass, and a volume

    number of objects = density * volume / mass
    """

    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, rescaling, kwargs_model, delta_power_law_index=0.0):
        """

        :param z:
        :param delta_z:
        :param geometry_class:
        :param rescaling:
        :param log_mlow:
        :param log_mhigh:
        :param draw_poisson:
        :param mass_function_model:
        :param m_pivot:
        :return:
        """
        m_pivot = 10 ** 8
        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 10)
        m_h = m * h
        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / m_pivot), np.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = 10 ** coeffs[1]
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = rescaling * norm_dv * volume_element_comoving
        kwargs_model['normalization'] = normalization
        kwargs_model['power_law_index'] = plaw_index
        return ShethTormenMixedWDM(**kwargs_model)

class ShethTormenTurnover(WDMPowerLaw):
    """
    This class generates masses from a delta function normalized with respect to a
    background density, a mass, and a volume

    number of objects = density * volume / mass
    """

    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, rescaling, kwargs_model, delta_power_law_index=0.0):
        """

        :param z:
        :param delta_z:
        :param geometry_class:
        :param rescaling:
        :param log_mlow:
        :param log_mhigh:
        :param draw_poisson:
        :param mass_function_model:
        :param m_pivot:
        :return:
        """
        m_pivot = 10**8
        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 10)
        m_h = m * h
        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / m_pivot), np.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = 10 ** coeffs[1]
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = rescaling * norm_dv * volume_element_comoving
        kwargs_model['normalization'] = normalization
        kwargs_model['power_law_index'] = plaw_index
        return ShethTormenTurnover(**kwargs_model)

class ShethTormenMixedWDM(MixedWDMPowerLaw):
    """
    This class generates masses from a delta function normalized with respect to a
    background density, a mass, and a volume

    number of objects = density * volume / mass
    """

    @classmethod
    def from_redshift(cls, z, delta_z, geometry_class, rescaling, kwargs_model, delta_power_law_index=0.0):
        """

        :param z:
        :param delta_z:
        :param geometry_class:
        :param rescaling:
        :param log_mlow:
        :param log_mhigh:
        :param draw_poisson:
        :param mass_function_model:
        :param m_pivot:
        :return:
        """
        m_pivot = 10 ** 8
        h = geometry_class.cosmo.h
        # To M_sun / h units
        m = np.logspace(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], 10)
        m_h = m * h
        dndlogM = massFunction(m_h, z, q_out='dndlnM', model='sheth99')
        dndM_comoving_h = dndlogM / m
        # three factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3
        coeffs = np.polyfit(np.log10(m / m_pivot), np.log10(dndM_comoving), 1)
        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = 10 ** coeffs[1]
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = rescaling * norm_dv * volume_element_comoving
        kwargs_model['normalization'] = normalization
        kwargs_model['power_law_index'] = plaw_index
        return ShethTormenMixedWDM(**kwargs_model)
