from copy import deepcopy
import numpy as np
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw, WDMPowerLaw, MixedWDMPowerLaw
from colossus.lss.mass_function import massFunction


class ShethTormen(CDMPowerLaw):
    """
    This class samples from the Sheth-Tormen halo mass function
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
        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = 10 ** coeffs[1] / (m_pivot**plaw_index)
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = rescaling * norm_dv * volume_element_comoving

        return ShethTormen(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                           kwargs_model['draw_poisson'], normalization)

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
        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = (10 ** coeffs[1]) / (m_pivot**plaw_index)
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = rescaling * norm_dv * volume_element_comoving

        return ShethTormenTurnover(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                           kwargs_model['draw_poisson'], normalization, kwargs_model['log_mc'],
                                   kwargs_model['a_wdm'], kwargs_model['b_wdm'], kwargs_model['c_wdm'])

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
        plaw_index = coeffs[0] + delta_power_law_index
        norm_dv = 10 ** coeffs[1] / (m_pivot**plaw_index)
        volume_element_comoving = geometry_class.volume_element_comoving(z, delta_z)
        normalization = rescaling * norm_dv * volume_element_comoving
        return ShethTormenMixedWDM(kwargs_model['log_mlow'], kwargs_model['log_mhigh'], plaw_index,
                                         kwargs_model['draw_poisson'], normalization, kwargs_model['log_mc'],
                                   kwargs_model['a_wdm'], kwargs_model['b_wdm'],
                                   kwargs_model['c_wdm'], kwargs_model['mixed_DM_frac'])

    # def _setup_colossus_cosmology(self, astropy_instance):
    #
    #     if not hasattr(self, 'colossus_cosmo'):
    #         colossus_kwargs = {}
    #         colossus_kwargs['H0'] = astropy_instance.h * 100
    #         colossus_kwargs['Om0'] = astropy_instance.Om0
    #         colossus_kwargs['Ob0'] = astropy_instance.Ob0
    #         colossus_kwargs['ns'] = astropy_instance.ns
    #         colossus_kwargs['sigma8'] = astropy_instance.sigma8
    #         colossus_kwargs['power_law'] = False
    #         self._colossus_cosmo = colossus_cosmology.setCosmology('custom', colossus_kwargs)
    #     return self._colossus_cosmo
