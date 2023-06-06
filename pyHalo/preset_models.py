"""
These functions define some preset halo mass function models that can be specified instead of individually specifying
a bunch of keyword arguments. This is intended for a quick and easy user interface. The definitions shown here can also
act as a how-to guide if one wants to explore more complicated descriptions of the halo mass function, as the models
presented here show what each keyword argument accepted by pyHalo does.
"""
from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw
from pyHalo.Halos.HaloModels.TNFWemulator import TNFWSubhaloEmulator
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform
from pyHalo.Rendering.SpatialDistributions.nfw import ProjectedNFW
from pyHalo.truncation_models import truncation_models
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.mass_function_models import preset_mass_function_models
from pyHalo.single_realization import Realization
from copy import copy
import numpy as np
from pyHalo.realization_extensions import RealizationExtensions
from pyHalo.utilities import de_broglie_wavelength, MinHaloMassULDM

__all__ = ['preset_model_from_name', 'CDM', 'WDM', 'ULDM', 'SIDM_core_collapse', 'WDM_mixed',
           'CDMFromEmulator']

def preset_model_from_name(name):
    """
    Retruns a preset_model function from a string
    :param name: the name of the preset model, should be the name of a function in this file
    :return: the function
    """
    if name == 'CDM':
        return CDM
    elif name == 'WDM':
        return WDM
    elif name == 'SIDM_core_collapse':
        return SIDM_core_collapse
    elif name == 'ULDM':
        return ULDM
    elif name == 'CDMEmulator':
        return CDMFromEmulator
    elif name == 'WDM_mixed':
        return WDM_mixed
    else:
        raise Exception('preset model '+ str(name)+' not recognized!')

def CDM(z_lens, z_source, sigma_sub=0.025, log_mlow=6., log_mhigh=10.,
        concentration_model_subhalos='DIEMERJOYCE19', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='DIEMERJOYCE19', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_ROCHE_GILMAN2020', kwargs_truncation_model_subhalos={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_m_host=13.3,  r_tidal=0.25,
        LOS_normalization=1.0, two_halo_contribution=True, delta_power_law_index=0.0,
        geometry_type='DOUBLE_CONE', kwargs_cosmo=None):
    """
    This class generates realizations of dark matter structure in Cold Dark Matter

    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param sigma_sub: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2]
    :param log_mlow: log base 10 of the minimum halo mass to render
    :param log_mhigh: log base 10 of the maximum halo mass to render
    :param concentration_model_subhalos: the concentration-mass relation applied to subhalos,
    see concentration_models.py for a complete list of available models
    :param kwargs_concentration_model_subhalos: keyword arguments for the subhalo MC relation
    NOTE: keyword args returned by the load_concentration_model override these keywords with duplicate arguments
    :param concentration_model_subhalos: the concentration-mass relation applied to field halos,
    see concentration_models.py for a complete list of available models
    :param kwargs_concentration_model_fieldhalos: keyword arguments for the field halo MC relation
    NOTE: keyword args returned by the load_concentration_model override these keywords with duplicate arguments
    :param truncation_model_subhalos: the truncation model applied to subhalos, see truncation_models for a complete list
    :param kwargs_truncation_model_subhalos: keyword arguments for the truncation model applied to subhalos
    :param truncation_model_fieldhalos: the truncation model applied to field halos, see truncation_models for a
    complete list
    :param kwargs_truncation_model_fieldhalos: keyword arguments for the truncation model applied to field halos
    :param shmf_log_slope: the logarithmic slope of the subhalo mass function pivoting around 10^8 M_sun
    :param cone_opening_angle_arcsec: the opening angle of the rendering volume in arcsec
    :param log_m_host: log base 10 of the host halo mass [M_sun]
    :param r_tidal: the core radius of the host halo in units of the host halo scale radius. Subhalos are distributed
    in 3D with a cored NFW profile with this core radius
    :param LOS_normalization: rescales the amplitude of the line-of-sight halo mass function
    :param two_halo_contribution: bool; turns on and off the two-halo term
    :param delta_power_law_index: tilts the logarithmic slope of the subhalo and field halo mass functions around pivot
    at 10^8 M_sun
    :param geometry_type: string that specifies the geometry of the rendering volume; options include
    DOUBLE_CONE, CONE, CYLINDER
    :param kwargs_cosmo: keyword arguments that specify the cosmology (see pyHalo.Cosmology.cosmology)
    :return: a realization of dark matter halos
    """

    # FIRST WE CREATE AN INSTANCE OF PYHALO, WHICH SETS THE COSMOLOGY
    pyhalo = pyHalo(z_lens, z_source, kwargs_cosmo)
    # WE ALSO SPECIFY THE GEOMETRY OF THE RENDERING VOLUME
    geometry = Geometry(pyhalo.cosmology, z_lens, z_source,
                        cone_opening_angle_arcsec, geometry_type)

    # NOW WE SET THE MASS FUNCTION CLASSES FOR SUBHALOS AND FIELD HALOS
    # NOTE: MASS FUNCTION CLASSES SHOULD NOT BE INSTANTIATED HERE
    mass_function_model_subhalos = CDMPowerLaw
    mass_function_model_fieldhalos = ShethTormen

    # SET THE SPATIAL DISTRIBUTION MODELS FOR SUBHALOS AND FIELD HALOS:
    subhalo_spatial_distribution = ProjectedNFW
    fieldhalo_spatial_distribution = LensConeUniform

    # set the density profile definition
    mdef_subhalos = 'TNFW'
    mdef_field_halos = 'TNFW'

    kwargs_concentration_model_subhalos['cosmo'] = pyhalo.astropy_cosmo
    kwargs_concentration_model_fieldhalos['cosmo'] = pyhalo.astropy_cosmo

    # SET THE CONCENTRATION-MASS RELATION FOR SUBHALOS AND FIELD HALOS
    model_subhalos, kwargs_mc_subs = preset_concentration_models(concentration_model_subhalos,
                                                                 kwargs_concentration_model_subhalos)
    concentration_model_subhalos = model_subhalos(**kwargs_mc_subs)

    model_fieldhalos, kwargs_mc_field = preset_concentration_models(concentration_model_fieldhalos,
                                                                    kwargs_concentration_model_fieldhalos)
    concentration_model_fieldhalos = model_fieldhalos(**kwargs_mc_field)

    # SET THE TRUNCATION RADIUS FOR SUBHALOS AND FIELD HALOS
    kwargs_truncation_model_subhalos['lens_cosmo'] = pyhalo.lens_cosmo
    kwargs_truncation_model_fieldhalos['lens_cosmo'] = pyhalo.lens_cosmo

    model_subhalos, kwargs_trunc_subs = truncation_models(truncation_model_subhalos)
    kwargs_trunc_subs.update(kwargs_truncation_model_subhalos)
    truncation_model_subhalos = model_subhalos(**kwargs_trunc_subs)

    model_fieldhalos, kwargs_trunc_field = truncation_models(truncation_model_fieldhalos)
    kwargs_trunc_field.update(kwargs_truncation_model_fieldhalos)
    truncation_model_fieldhalos = model_fieldhalos(**kwargs_trunc_field)

    # NOW THAT THE CLASSES ARE SPECIFIED, WE SORT THE KEYWORD ARGUMENTS AND CLASSES INTO LISTS
    population_model_list = ['SUBHALOS', 'LINE_OF_SIGHT']
    mass_function_class_list = [mass_function_model_subhalos, mass_function_model_fieldhalos]
    kwargs_subhalos = {'log_mlow': log_mlow,
                       'log_mhigh': log_mhigh,
                       'm_pivot': 10 ** 8,
                       'power_law_index': shmf_log_slope,
                       'delta_power_law_index': delta_power_law_index,
                       'log_m_host': log_m_host,
                       'sigma_sub': sigma_sub}
    kwargs_los = {'log_mlow': log_mlow,
                       'log_mhigh': log_mhigh,
                  'LOS_normalization': LOS_normalization,
                       'm_pivot': 10 ** 8,
                       'delta_power_law_index': delta_power_law_index,
                       'log_m_host': log_m_host}
    kwargs_mass_function_list = [kwargs_subhalos, kwargs_los]
    spatial_distribution_class_list = [subhalo_spatial_distribution, fieldhalo_spatial_distribution]
    kwargs_subhalos_spatial = {'m_host': 10 ** log_m_host, 'zlens': z_lens,
                               'rmax2d_arcsec': cone_opening_angle_arcsec / 2, 'r_core_units_rs': r_tidal,
                               'lens_cosmo': pyhalo.lens_cosmo}
    kwargs_los_spatial = {'cone_opening_angle': cone_opening_angle_arcsec, 'geometry': geometry}
    kwargs_spatial_distribution_list = [kwargs_subhalos_spatial, kwargs_los_spatial]

    if two_halo_contribution:
        population_model_list += ['TWO_HALO']
        kwargs_two_halo = copy(kwargs_los)
        kwargs_mass_function_list += [kwargs_two_halo]
        spatial_distribution_class_list += [LensConeUniform]
        kwargs_spatial_distribution_list += [kwargs_los_spatial]
        mass_function_class_list += [ShethTormen]

    kwargs_halo_model = {'truncation_model_subhalos': truncation_model_subhalos,
                         'concentration_model_subhalos': concentration_model_subhalos,
                         'truncation_model_field_halos': truncation_model_fieldhalos,
                         'concentration_model_field_halos': concentration_model_fieldhalos,
                         'kwargs_density_profile': {}}

    realization_list = pyhalo.render(population_model_list, mass_function_class_list, kwargs_mass_function_list,
                                          spatial_distribution_class_list, kwargs_spatial_distribution_list,
                                          geometry, mdef_subhalos, mdef_field_halos, kwargs_halo_model, nrealizations=1)
    return realization_list[0]

def WDM(z_lens, z_source, log_mc, sigma_sub=0.025, log_mlow=6., log_mhigh=10.,
        mass_function_model_subhalos='LOVELL2020', kwargs_mass_function_subhalos={},
        mass_function_model_fieldhalos='LOVELL2020', kwargs_mass_function_fieldhalos={},
        concentration_model_subhalos='BOSE2016', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='BOSE2016', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_ROCHE_GILMAN2020', kwargs_truncation_model_subhalos={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_m_host=13.3, r_tidal=0.25,
        LOS_normalization=1.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None,
        mdef_subhalos='TNFW', mdef_field_halos='TNFW', kwargs_density_profile={}):

    """
    This class generates realizations of dark matter structure in Warm Dark Matter

    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param log_mc: the log base 10 of the half-mode mass
    :param sigma_sub: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2]
    :param log_mlow: log base 10 of the minimum halo mass to render
    :param log_mhigh: log base 10 of the maximum halo mass to render
    :param mass_function_model_subhalos: mass function model for subhalos, see mass_function_models.py for a list
    :param kwargs_mass_function_subhalos: keyword arguments for the mass function model
    :param mass_function_model_fieldhalos: mass function model for field halos, see mass_function_models.py for a list
    :param kwargs_mass_function_fieldhalos: keyword arguments for the mass function model
    :param concentration_model_subhalos: the concentration-mass relation applied to subhalos,
    see concentration_models.py for a complete list of available models
    :param kwargs_concentration_model_subhalos: keyword arguments for the subhalo MC relation
    NOTE: keyword args returned by the load_concentration_model override these keywords with duplicate arguments
    :param concentration_model_subhalos: the concentration-mass relation applied to field halos,
    see concentration_models.py for a complete list of available models
    :param kwargs_concentration_model_fieldhalos: keyword arguments for the field halo MC relation
    NOTE: keyword args returned by the load_concentration_model override these keywords with duplicate arguments
    :param truncation_model_subhalos: the truncation model applied to subhalos, see truncation_models for a complete list
    :param kwargs_truncation_model_subhalos: keyword arguments for the truncation model applied to subhalos
    :param truncation_model_fieldhalos: the truncation model applied to field halos, see truncation_models for a
    complete list
    :param kwargs_truncation_model_fieldhalos: keyword arguments for the truncation model applied to field halos
    :param shmf_log_slope: the logarithmic slope of the subhalo mass function pivoting around 10^8 M_sun
    :param cone_opening_angle_arcsec: the opening angle of the rendering volume in arcsec
    :param log_m_host: log base 10 of the host halo mass [M_sun]
    :param r_tidal: the core radius of the host halo in units of the host halo scale radius. Subhalos are distributed
    in 3D with a cored NFW profile with this core radius
    :param geometry_type: string that specifies the geometry of the rendering volume; options include
    DOUBLE_CONE, CONE, CYLINDER
    :param kwargs_cosmo: keyword arguments that specify the cosmology (see pyHalo.Cosmology.cosmology)
    :param mdef_subhalos: mass definition for subhalos
    :param mdef_field_halos: mass definition for field halos
    :param kwargs_density_profile: keyword arguments for the specified mass profile
    :return: a realization of dark matter halos
    """
    # FIRST WE CREATE AN INSTANCE OF PYHALO, WHICH SETS THE COSMOLOGY
    pyhalo = pyHalo(z_lens, z_source, kwargs_cosmo)
    # WE ALSO SPECIFY THE GEOMETRY OF THE RENDERING VOLUME
    geometry = Geometry(pyhalo.cosmology, z_lens, z_source,
                        cone_opening_angle_arcsec, geometry_type)

    # SET THE SPATIAL DISTRIBUTION MODELS FOR SUBHALOS AND FIELD HALOS:
    subhalo_spatial_distribution = ProjectedNFW
    fieldhalo_spatial_distribution = LensConeUniform

    # SET THE MASS FUNCTION MODELS FOR SUBHALOS AND FIELD HALOS
    # NOTE: MASS FUNCTIONS SHOULD NOT BE INSTANTIATED HERE
    mass_function_model_subhalos, kwargs_mfunc_subs = preset_mass_function_models(mass_function_model_subhalos)
    kwargs_mfunc_subs.update(kwargs_mass_function_subhalos)
    kwargs_mfunc_subs['log_mc'] = log_mc

    mass_function_model_fieldhalos, kwargs_mfunc_field = preset_mass_function_models(mass_function_model_fieldhalos)
    kwargs_mfunc_field.update(kwargs_mass_function_fieldhalos)
    kwargs_mfunc_field['log_mc'] = log_mc

    # SET THE CONCENTRATION-MASS RELATION FOR SUBHALOS AND FIELD HALOS
    kwargs_concentration_model_subhalos['cosmo'] = pyhalo.astropy_cosmo
    kwargs_concentration_model_subhalos['log_mc'] = log_mc
    kwargs_concentration_model_fieldhalos['cosmo'] = pyhalo.astropy_cosmo
    kwargs_concentration_model_fieldhalos['log_mc'] = log_mc

    model_subhalos, kwargs_mc_subs = preset_concentration_models(concentration_model_subhalos,
                                                                 kwargs_concentration_model_subhalos)
    concentration_model_CDM = preset_concentration_models('DIEMERJOYCE19')[0]
    kwargs_mc_subs['concentration_cdm_class'] = concentration_model_CDM
    concentration_model_subhalos = model_subhalos(**kwargs_mc_subs)

    model_fieldhalos, kwargs_mc_field = preset_concentration_models(concentration_model_fieldhalos,
                                                                    kwargs_concentration_model_fieldhalos)
    kwargs_mc_field['cosmo'] = pyhalo.astropy_cosmo
    kwargs_mc_field['log_mc'] = log_mc
    concentration_model_CDM = preset_concentration_models('DIEMERJOYCE19')[0]
    kwargs_mc_field['concentration_cdm_class'] = concentration_model_CDM
    concentration_model_fieldhalos = model_fieldhalos(**kwargs_mc_field)

    # SET THE TRUNCATION RADIUS FOR SUBHALOS AND FIELD HALOS
    kwargs_truncation_model_subhalos['lens_cosmo'] = pyhalo.lens_cosmo
    kwargs_truncation_model_fieldhalos['lens_cosmo'] = pyhalo.lens_cosmo

    model_subhalos, kwargs_trunc_subs = truncation_models(truncation_model_subhalos)
    kwargs_trunc_subs.update(kwargs_truncation_model_subhalos)
    kwargs_trunc_subs['lens_cosmo'] = pyhalo.lens_cosmo
    truncation_model_subhalos = model_subhalos(**kwargs_trunc_subs)

    model_fieldhalos, kwargs_trunc_field = truncation_models(truncation_model_fieldhalos)
    kwargs_trunc_field.update(kwargs_truncation_model_fieldhalos)
    kwargs_trunc_field['lens_cosmo'] = pyhalo.lens_cosmo
    truncation_model_fieldhalos = model_fieldhalos(**kwargs_trunc_field)

    # NOW THAT THE CLASSES ARE SPECIFIED, WE SORT THE KEYWORD ARGUMENTS AND CLASSES INTO LISTS
    population_model_list = ['SUBHALOS', 'LINE_OF_SIGHT', 'TWO_HALO']

    mass_function_class_list = [mass_function_model_subhalos,
                                mass_function_model_fieldhalos,
                                mass_function_model_fieldhalos]
    kwargs_mfunc_subs.update({'log_mlow': log_mlow,
                       'log_mhigh': log_mhigh,
                       'm_pivot': 10 ** 8,
                       'power_law_index': shmf_log_slope,
                       'log_m_host': log_m_host,
                       'sigma_sub': sigma_sub,
                       'delta_power_law_index': 0.0})
    kwargs_mfunc_field.update({'log_mlow': log_mlow,
                  'log_mhigh': log_mhigh,
                  'LOS_normalization': LOS_normalization,
                  'm_pivot': 10 ** 8,
                  'log_m_host': log_m_host,
                  'delta_power_law_index': 0.0})
    kwargs_mass_function_list = [kwargs_mfunc_subs, kwargs_mfunc_field, kwargs_mfunc_field]
    spatial_distribution_class_list = [subhalo_spatial_distribution, fieldhalo_spatial_distribution, fieldhalo_spatial_distribution]
    kwargs_subhalos_spatial = {'m_host': 10 ** log_m_host, 'zlens': z_lens,
                               'rmax2d_arcsec': cone_opening_angle_arcsec / 2, 'r_core_units_rs': r_tidal,
                               'lens_cosmo': pyhalo.lens_cosmo}
    kwargs_los_spatial = {'cone_opening_angle': cone_opening_angle_arcsec, 'geometry': geometry}
    kwargs_spatial_distribution_list = [kwargs_subhalos_spatial, kwargs_los_spatial, kwargs_los_spatial]

    kwargs_halo_model = {'truncation_model_subhalos': truncation_model_subhalos,
                         'concentration_model_subhalos': concentration_model_subhalos,
                         'truncation_model_field_halos': truncation_model_fieldhalos,
                         'concentration_model_field_halos': concentration_model_fieldhalos,
                         'kwargs_density_profile': kwargs_density_profile}

    realization_list = pyhalo.render(population_model_list, mass_function_class_list, kwargs_mass_function_list,
                                     spatial_distribution_class_list, kwargs_spatial_distribution_list,
                                     geometry, mdef_subhalos, mdef_field_halos, kwargs_halo_model, nrealizations=1)
    return realization_list[0]

def ULDM(z_lens, z_source, log10_m_uldm, log10_fluc_amplitude=-0.8, fluctuation_size_scale=0.05,
          fluctuation_size_dispersion=0.2, n_fluc_scale=1.0, velocity_scale=200, sigma_sub=0.025, log_mlow=6., log_mhigh=10.,
        mass_function_model_subhalos='SHMF_SCHIVE2016', kwargs_mass_function_subhalos={},
        mass_function_model_fieldhalos='SCHIVE2016', kwargs_mass_function_fieldhalos={},
        concentration_model_subhalos='LAROCHE2022', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='LAROCHE2022', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_ROCHE_GILMAN2020', kwargs_truncation_model_subhalos={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_m_host=13.3, r_tidal=0.25,
        LOS_normalization=1.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None,
         uldm_plaw=1 / 3, flucs=True, flucs_shape='aperture', flucs_args={}, n_cut=50000, r_ein=1.0):

    """
    This generates realizations of ultra-light dark matter (ULDM), including the ULDM halo mass function and halo density profiles,
    as well as density fluctuations in the main deflector halo.

    Similarly to WDMGeneral, the functional form of the subhalo mass function is the same as the field halo mass function.
    However, this model differs from WDMGeneral by creating halos which are composite ULDM + NFW density profiles.
    The ULDM particle mass and core radius-halo mass power law exponent must now be specified. For details regarding ULDM halos,
    see Schive et al. 2014 (https://arxiv.org/pdf/1407.7762.pdf). Equations (3) and (7) give the soliton density profile
    and core radius, respectively.

    The differential halo mass function is described by three parameters, see Schive et al. 2016
    (https://arxiv.org/pdf/1508.04621.pdf):

    1) log_m0 - the log10 value of the characteristic mass, or the scale where the ULDM mass function begins
    to deviate from CDM:

        m0 = (1.6*10**10) * (m22)**(-4/3)   [solar masses],

    where

        m22 = m_uldm / 10**22 eV.

    2) b_uldm - modifies the log slope of the ULDM mass function, analogous to b_wdm (see WDMLovell2020)

    3) c_uldm - modifies the log slope of the ULDM mass function, analogous to c_wdm (see WDMLovell2020)

    The parametrization for the mass function is:

        n_uldm / n_cdm = (1 + (m / m_0)^b_uldm) ^ c_uldm,

    where Schive et al. 2016 determined that

        (m_0,    b_uldm,    c_uldm) = ( (1.6*10**10) * (m22)**(-4/3),    1.1,    2.2)

    As for the concentration relative to CDM, there hasa not been a detailed study to date. Hoever, since the formation
    history and collapse time determines concentration, we can reasonably use a WDM concentration-mass relation
    such as the Lovell 2020 formula,
    i.e. simply c_wdm = c_uldm, with (c_scale, c_power) = (15, -0.3).
    or the Bose et al. formula with (c_scale, c_power) = (60, -0.17) and a very negligible redshift dependence.

    The default model was computed using the formalism presented by Schneider et al. (2015) using a ULDM power spectrum,
    and results in a sharper cutoff with (c_scale, c_power, c_power_inner) = (3.348, -0.489, 1.5460)

    The form for the ULDM concentration-mass relation turnover is (1 + c_scale * (mhm/m)^c_power_inner)^c_power

    Furthermore, Schive et al. 2014 (https://arxiv.org/pdf/1407.7762.pdf) found a redshift-dependent minimum ULDM halo mass
    given by

        M_min(z) = a^(-3/4) * (Zeta(z)/Zeta(0))^(1/4) * M_min,0     [solar masses]

    where a is the cosmic scale factor,

        M_min,0 = 4.4*10^7 * m22^(-3/2)     [solar masses]

    and

        Zeta(z) = (18*pi^2 + 82(Om(z) - 1) - 39(Om(z) - 1)^2) / Om(z),

    where Om(z) is the matter density parameter.

    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param log10_m_uldm: log base 10 of the ULDM particle mass
    :param log10_fluc_amplitude: log base 10 of the fluctuation amplitude in the host halo density profile
    :param fluctuation_size_scale: sets the size of the fluctuations in host halo in units of de-Broglie wavelength
    :param fluctuation_size_dispersion: sets the variance of the size of the fluctuations
    :param n_fluc_scale: rescales the number of fluctuations
    :param velocity_scale: the velocity scale of the host halo used to convert particle mass to de-Broglie wavelength
    :param sigma_sub: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2]
    :param log_mlow: log base 10 of the minimum halo mass to render
    :param log_mhigh: log base 10 of the maximum halo mass to render
    :param mass_function_model_subhalos: mass function model for subhalos, see mass_function_models.py for a list
    :param kwargs_mass_function_subhalos: keyword arguments for the mass function model
    :param mass_function_model_fieldhalos: mass function model for field halos, see mass_function_models.py for a list
    :param kwargs_mass_function_fieldhalos: keyword arguments for the mass function model
    :param concentration_model_subhalos: the concentration-mass relation applied to subhalos,
    see concentration_models.py for a complete list of available models
    :param kwargs_concentration_model_subhalos: keyword arguments for the subhalo MC relation
    NOTE: keyword args returned by the load_concentration_model override these keywords with duplicate arguments
    :param concentration_model_subhalos: the concentration-mass relation applied to field halos,
    see concentration_models.py for a complete list of available models
    :param kwargs_concentration_model_fieldhalos: keyword arguments for the field halo MC relation
    NOTE: keyword args returned by the load_concentration_model override these keywords with duplicate arguments
    :param truncation_model_subhalos: the truncation model applied to subhalos, see truncation_models for a complete list
    :param kwargs_truncation_model_subhalos: keyword arguments for the truncation model applied to subhalos
    :param truncation_model_fieldhalos: the truncation model applied to field halos, see truncation_models for a
    complete list
    :param kwargs_truncation_model_fieldhalos: keyword arguments for the truncation model applied to field halos
    :param shmf_log_slope: the logarithmic slope of the subhalo mass function pivoting around 10^8 M_sun
    :param cone_opening_angle_arcsec: the opening angle of the rendering volume in arcsec
    :param log_m_host: log base 10 of the host halo mass [M_sun]
    :param r_tidal: the core radius of the host halo in units of the host halo scale radius. Subhalos are distributed
    in 3D with a cored NFW profile with this core radius
    :param geometry_type: string that specifies the geometry of the rendering volume; options include
    DOUBLE_CONE, CONE, CYLINDER
    :param kwargs_cosmo: keyword arguments that specify the cosmology, see Cosmology/cosmology.py
    :param uldm_plaw: sets the exponent for the core size/halo mass relation in ULDM
    :param flucs: bool; turns of/off fluctuations in the host halo density profile
    :param flucs_shape: the geometry in which to render fluctuations, options include 'aperture', 'ring', 'ellipse'
    :param flucs_args: keyword arguments for the specified geometry, see realization_extensions
    :param n_cut: maximum number of fluctuations to render; amplitudes of individual fluctuations get suppressed by
    sqrt(n_cut / n_total) if n_total > n_cut, where n_total is the number of expected fluctuations given the de-Broglie
    wavelength and the area in which the fluctuations area rendered
    :param r_ein: The characteristic angular used to convert fluctuations in convergence to fluctuations in mass
    :return: a realization of halos and fluctuations modeled as Gaussians in ULDM
    """

    # constants
    m22 = 10**(log10_m_uldm + 22)
    log_m0 = np.log10(1.6e10 * m22**(-4/3))

    # FIRST WE CREATE AN INSTANCE OF PYHALO, WHICH SETS THE COSMOLOGY
    pyhalo = pyHalo(z_lens, z_source, kwargs_cosmo)

    #compute M_min as described in documentation
    log_m_min = MinHaloMassULDM(log10_m_uldm, pyhalo.astropy_cosmo, log_mlow)
    kwargs_density_profile = {}
    kwargs_density_profile['log10_m_uldm'] = log10_m_uldm
    kwargs_density_profile['scale_nfw'] = False
    kwargs_density_profile['uldm_plaw'] = uldm_plaw
    kwargs_wdm = {'z_lens': z_lens, 'z_source': z_source, 'log_mc': log_m0, 'sigma_sub': sigma_sub,
                  'log_mlow': log_m_min, 'log_mhigh': log_mhigh,
                  'mass_function_model_subhalos': mass_function_model_subhalos,
                  'kwargs_mass_function_subhalos': kwargs_mass_function_subhalos,
                  'mass_function_model_fieldhalos': mass_function_model_fieldhalos,
                  'kwargs_mass_function_fieldhalos': kwargs_mass_function_fieldhalos,
                  'concentration_model_subhalos': concentration_model_subhalos,
                  'kwargs_concentration_model_subhalos': kwargs_concentration_model_subhalos,
                  'concentration_model_fieldhalos': concentration_model_fieldhalos,
                  'kwargs_concentration_model_fieldhalos': kwargs_concentration_model_fieldhalos,
                  'truncation_model_subhalos': truncation_model_subhalos,
                  'kwargs_truncation_model_subhalos': kwargs_truncation_model_subhalos,
                  'truncation_model_fieldhalos': truncation_model_fieldhalos,
                  'kwargs_truncation_model_fieldhalos': kwargs_truncation_model_fieldhalos,
                  'shmf_log_slope': shmf_log_slope, 'cone_opening_angle_arcsec': cone_opening_angle_arcsec,
                  'log_m_host': log_m_host, 'r_tidal': r_tidal, 'LOS_normalization': LOS_normalization,
                  'geometry_type': geometry_type, 'kwargs_cosmo': kwargs_cosmo,
                  'mdef_subhalos': 'ULDM', 'mdef_field_halos': 'ULDM',
                  'kwargs_density_profile': kwargs_density_profile
                  }

    uldm_no_fluctuations = WDM(**kwargs_wdm)

    if flucs: # add fluctuations to realization
        ext = RealizationExtensions(uldm_no_fluctuations)
        lambda_dB = de_broglie_wavelength(log10_m_uldm, velocity_scale) # de Broglie wavelength in kpc

        if flucs_args=={}:
            raise Exception('Must specify fluctuation arguments, see realization_extensions.add_ULDM_fluctuations')

        a_fluc = 10 ** log10_fluc_amplitude
        m_psi = 10 ** log10_m_uldm

        zlens_ref, zsource_ref = 0.5, 2.0
        mhost_ref = 10**13.3
        rein_ref = 1.0
        r_perp_ref = rein_ref * uldm_no_fluctuations.lens_cosmo.cosmo.kpc_proper_per_asec(zlens_ref)

        model, _ = preset_concentration_models('DIEMERJOYCE19')
        concentration_model_for_host = model(pyhalo.astropy_cosmo)
        sigma_crit_ref = uldm_no_fluctuations.lens_cosmo.get_sigma_crit_lensing(zlens_ref, zsource_ref)
        c_host_ref = concentration_model_for_host.nfw_concentration(mhost_ref, z_lens)
        rhos_ref, rs_ref, _ = uldm_no_fluctuations.lens_cosmo.NFW_params_physical(mhost_ref, c_host_ref, zlens_ref)
        xref = r_perp_ref/rs_ref
        if xref < 1:
            Fxref = np.arctanh(np.sqrt(1 - xref ** 2)) / np.sqrt(1 - xref ** 2)
        else:
            Fxref = np.arctan(np.sqrt(-1 + xref ** 2)) / np.sqrt(-1 + xref ** 2)
        sigma_host_ref = 2 * rhos_ref * rs_ref * (1-Fxref)/(xref**2 - 1)

        r_perp = r_ein * uldm_no_fluctuations.lens_cosmo.cosmo.kpc_proper_per_asec(z_lens)
        sigma_crit = uldm_no_fluctuations.lens_cosmo.get_sigma_crit_lensing(z_lens, z_source)
        c_host = concentration_model_for_host.nfw_concentration(10**log_m_host, z_lens)
        rhos, rs, _ = uldm_no_fluctuations.lens_cosmo.NFW_params_physical(10**log_m_host, c_host, z_lens)
        x = r_perp / rs
        if x < 1:
            Fx = np.arctanh(np.sqrt(1 - x ** 2)) / np.sqrt(1 - x ** 2)
        else:
            Fx = np.arctan(np.sqrt(-1 + x ** 2)) / np.sqrt(-1 + x ** 2)
        sigma_host = 2 * rhos * rs * (1 - Fx) / (x ** 2 - 1)

        fluctuation_amplitude = a_fluc * (m_psi / 1e-22) ** -0.5 * \
                                (sigma_crit_ref/sigma_crit) * (sigma_host/sigma_host_ref)

        uldm_realization = ext.add_ULDM_fluctuations(de_Broglie_wavelength=lambda_dB,
                                fluctuation_amplitude=fluctuation_amplitude,
                                fluctuation_size=lambda_dB * fluctuation_size_scale,
                                fluctuation_size_variance=lambda_dB * fluctuation_size_scale * fluctuation_size_dispersion,
                                n_fluc_scale=n_fluc_scale,
                                shape=flucs_shape,
                                args=flucs_args,
                                n_cut=n_cut)
        return uldm_realization
    else:
        return uldm_no_fluctuations

def SIDM_core_collapse(z_lens, z_source, mass_ranges_subhalos, mass_ranges_field_halos,
        probabilities_subhalos, probabilities_field_halos, kwargs_sub_function=None,
        kwargs_field_function=None, sigma_sub=0.025, log_mlow=6., log_mhigh=10.,
        concentration_model_subhalos='DIEMERJOYCE19', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='DIEMERJOYCE19', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_ROCHE_GILMAN2020', kwargs_truncation_model_subhalos={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_m_host=13.3,  r_tidal=0.25,
        LOS_normalization=1.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None, collapsed_halo_profile='SPL_CORE',
        kwargs_collapsed_profile={'x_core_halo': 0.05, 'x_match': 3.0, 'log_slope_halo': 3.0}):

    """
    Generates realizations of SIDM given the fraction of core-collapsed halos as a function of halo mass

    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param mass_ranges_subhalos: a list of mass ranges for subhalo collapse, e.g.
    mass_ranges_subhalos = [[6, 8], [8, 9], ...]
    :param mass_ranges_field_halos: a list of mass ranges for subhalo collapse, e.g.
    mass_ranges_field_halos = [[6, 8], [8, 9], ...]
    :param probabilities_subhalos: a list of functions, or a list of the fraction of collapsed subhalos in each mass bin
    :param probabilities_field_halos: a list of functions, or a list of the fraction of collapsed field halos in each
    mass bin
    :param kwargs_sub_function: if probabilities_subhalos is a list of functions, specifies keyword arguments
    for each function
    :param kwargs_field_function: if probabilities_field_halos is a list of functions, specifies keyword arguments
    for each function
    :param sigma_sub: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2]
    :param log_mlow: log base 10 of the minimum halo mass to render
    :param log_mhigh: log base 10 of the maximum halo mass to render
    :param concentration_model_subhalos: the concentration-mass relation applied to subhalos,
    see concentration_models.py for a complete list of available models
    :param kwargs_concentration_model_subhalos: keyword arguments for the subhalo MC relation
    NOTE: keyword args returned by the load_concentration_model override these keywords with duplicate arguments
    :param concentration_model_subhalos: the concentration-mass relation applied to field halos,
    see concentration_models.py for a complete list of available models
    :param kwargs_concentration_model_fieldhalos: keyword arguments for the field halo MC relation
    NOTE: keyword args returned by the load_concentration_model override these keywords with duplicate arguments
    :param truncation_model_subhalos: the truncation model applied to subhalos, see truncation_models for a complete list
    :param kwargs_truncation_model_subhalos: keyword arguments for the truncation model applied to subhalos
    :param truncation_model_fieldhalos: the truncation model applied to field halos, see truncation_models for a
    complete list
    :param kwargs_truncation_model_fieldhalos: keyword arguments for the truncation model applied to field halos
    :param shmf_log_slope: the logarithmic slope of the subhalo mass function pivoting around 10^8 M_sun
    :param cone_opening_angle_arcsec: the opening angle of the rendering volume in arcsec
    :param log_m_host: log base 10 of the host halo mass [M_sun]
    :param r_tidal: the core radius of the host halo in units of the host halo scale radius. Subhalos are distributed
    in 3D with a cored NFW profile with this core radius
    :param LOS_normalization: rescales the amplitude of the line-of-sight mass function
    :param geometry_type: string that specifies the geometry of the rendering volume; options include
    DOUBLE_CONE, CONE, CYLINDER
    :param kwargs_cosmo: keyword arguments that specify the cosmology, see Cosmology/cosmology.py
    :param collapsed_halo_profile: string that sets the density profile of core-collapsed halos
    currently implemented models are SPL_CORE and GNFW (see example notebook)
    :param kwargs_collapsed_profile: keyword arguments for the collapsed profile (see example notebook)
    :return: a realization of dark matter structure in SIDM
    """

    two_halo_contribution = True
    delta_power_law_index = 0.0
    cdm = CDM(z_lens, z_source, sigma_sub, log_mlow, log_mhigh,
        concentration_model_subhalos, kwargs_concentration_model_subhalos,
        concentration_model_fieldhalos, kwargs_concentration_model_fieldhalos,
        truncation_model_subhalos, kwargs_truncation_model_subhalos,
        truncation_model_fieldhalos, kwargs_truncation_model_fieldhalos,
        shmf_log_slope, cone_opening_angle_arcsec, log_m_host,  r_tidal,
        LOS_normalization, two_halo_contribution, delta_power_law_index,
        geometry_type, kwargs_cosmo)

    extension = RealizationExtensions(cdm)
    index_collapsed = extension.core_collapse_by_mass(mass_ranges_subhalos, mass_ranges_field_halos,
        probabilities_subhalos, probabilities_field_halos, kwargs_sub_function, kwargs_field_function)
    sidm = extension.add_core_collapsed_halos(index_collapsed, collapsed_halo_profile, **kwargs_collapsed_profile)
    return sidm

def WDM_mixed(z_lens, z_source, log_mc, mixed_DM_frac, sigma_sub=0.025, log_mlow=6., log_mhigh=10.,
        mass_function_model_subhalos='SHMF_MIXED_WDM_TURNOVER', kwargs_mass_function_subhalos={},
        mass_function_model_fieldhalos='MIXED_WDM_TURNOVER', kwargs_mass_function_fieldhalos={},
        concentration_model_subhalos='BOSE2016', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='BOSE2016', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_ROCHE_GILMAN2020', kwargs_truncation_model_subhalos={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_m_host=13.3, r_tidal=0.25,
        LOS_normalization=1.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None,
        kwargs_density_profile={}):

    """
    Implements the mixed dark matter model presented by Keely et al. (2023)
    https://arxiv.org/pdf/2301.07265.pdf

    :param z_lens:
    :param z_source:
    :param log_mc:
    :param mixed_DM_frac: fraction of dark matter in CDM component
    :param sigma_sub:
    :param log_mlow:
    :param log_mhigh:
    :param mass_function_model_subhalos:
    :param kwargs_mass_function_subhalos:
    :param mass_function_model_fieldhalos:
    :param kwargs_mass_function_fieldhalos:
    :param concentration_model_subhalos:
    :param kwargs_concentration_model_subhalos:
    :param concentration_model_fieldhalos:
    :param kwargs_concentration_model_fieldhalos:
    :param truncation_model_subhalos:
    :param kwargs_truncation_model_subhalos:
    :param truncation_model_fieldhalos:
    :param kwargs_truncation_model_fieldhalos:
    :param shmf_log_slope:
    :param cone_opening_angle_arcsec:
    :param log_m_host:
    :param r_tidal:
    :param LOS_normalization:
    :param geometry_type:
    :param kwargs_cosmo:
    :param kwargs_density_profile:
    :return:
    """
    params = ['a_wdm', 'b_wdm', 'c_wdm']
    values_keeley_2023 = [0.5, 0.8, -3.0]
    for i, param in enumerate(params):
        if param not in kwargs_mass_function_subhalos.keys():
            kwargs_mass_function_subhalos[param] = values_keeley_2023[i]
        if param not in kwargs_mass_function_fieldhalos.keys():
            kwargs_mass_function_fieldhalos[param] = values_keeley_2023[i]
    kwargs_mass_function_subhalos['mixed_DM_frac'] = mixed_DM_frac
    kwargs_mass_function_fieldhalos['mixed_DM_frac'] = mixed_DM_frac
    kwargs_wdm = {'z_lens': z_lens, 'z_source': z_source, 'log_mc': log_mc, 'sigma_sub': sigma_sub,
                  'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                  'mass_function_model_subhalos': mass_function_model_subhalos,
                  'kwargs_mass_function_subhalos': kwargs_mass_function_subhalos,
                  'mass_function_model_fieldhalos': mass_function_model_fieldhalos,
                  'kwargs_mass_function_fieldhalos': kwargs_mass_function_fieldhalos,
                  'concentration_model_subhalos': concentration_model_subhalos,
                  'kwargs_concentration_model_subhalos': kwargs_concentration_model_subhalos,
                  'concentration_model_fieldhalos': concentration_model_fieldhalos,
                  'kwargs_concentration_model_fieldhalos': kwargs_concentration_model_fieldhalos,
                  'truncation_model_subhalos': truncation_model_subhalos,
                  'kwargs_truncation_model_subhalos': kwargs_truncation_model_subhalos,
                  'truncation_model_fieldhalos': truncation_model_fieldhalos,
                  'kwargs_truncation_model_fieldhalos': kwargs_truncation_model_fieldhalos,
                  'shmf_log_slope': shmf_log_slope, 'cone_opening_angle_arcsec': cone_opening_angle_arcsec,
                  'log_m_host': log_m_host, 'r_tidal': r_tidal, 'LOS_normalization': LOS_normalization,
                  'geometry_type': geometry_type, 'kwargs_cosmo': kwargs_cosmo,
                  'kwargs_density_profile': kwargs_density_profile
                  }
    return WDM(**kwargs_wdm)

def CDMFromEmulator(z_lens, z_source, emulator_input, kwargs_cdm):
    """
    This generates a realization of subhalos using an emulator of the semi-analytic modeling code Galacticus, and
     generates line-of-sight halos from a mass function parameterized as Sheth-Tormen.

    :param z_lens: main deflector redshift
    :param z_source: sourcee redshift
    :param emulator_input: either an array or a callable function

    if callable: a function that returns an array of
    1) subhalo masses at infall [M_sun]
    2) subhalo projected x position [kpc]
    3) subhalo projected y position [kpc]
    4) subhalo final_bound_mass [M_sun]
    5) subhalo concentrations at infall

    if not callable: an array with shape (N_subhalos 5) that contains masses, positions x, positions y, etc.

    Mass convention is m200 with respect to the critical density of the Universe at redshift Z_infall, where Z_infall is
    the infall redshift (not necessarily the redshift at the time of lensing).

    :param cone_opening_angle_arcsec: the opening angle of the double cone rendering volume in arcsec
    :param log_mlow: log10(minimum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param log_mhigh: log10(maximum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param kwargs_other: allows for additional keyword arguments to be specified when creating realization

    The following optional keywords specify a concentration-mass relation for field halos parameterized as a power law
    in peak height. If they are not set in the function call, pyHalo assumes a default concentration-mass relation from Diemer&Joyce
    :param c0: amplitude of the mass-concentration relation at 10^8
    :param log10c0: logarithmic amplitude of the mass-concentration relation at 10^8 (only if c0_mcrelation is None)
    :param beta: logarithmic slope of the mass-concentration-relation pivoting around 10^8
    :param zeta: modifies the redshift evolution of the mass-concentration-relation
    :param two_halo_contribution: whether to include the two-halo term for correlated structure near the main deflector
    :param kwargs_halo_mass_function: keyword arguments passed to the LensingMassFunction class
    (see Cosmology.lensing_mass_function)
    :return: a realization of CDM halos
    """

    # we create a realization of only line-of-sight halos by setting sigma_sub = 0.0
    kwargs_cdm['sigma_sub'] = 0.0
    cdm_halos_LOS = CDM(z_lens, z_source, **kwargs_cdm)
    # get lens_cosmo class from class containing LOS objects; note that this will work even if there are no LOS halos
    lens_cosmo = cdm_halos_LOS.lens_cosmo

    # now create subhalos from the specified properties using the TNFWSubhaloEmulator class
    halo_list = []
    if callable(emulator_input):
        subhalo_infall_masses, subhalo_x_kpc, subhalo_y_kpc, subhalo_final_bound_masses, \
        subhalo_infall_concentrations = emulator_input()
    else:
        subhalo_infall_masses = emulator_input[:, 0]
        subhalo_x_kpc = emulator_input[:, 1]
        subhalo_y_kpc = emulator_input[:, 2]
        subhalo_final_bound_masses = emulator_input[:, 3]
        subhalo_infall_concentrations = emulator_input[:, 4]

    for i in range(0, len(subhalo_infall_masses)):
        halo = TNFWSubhaloEmulator(subhalo_infall_masses[i],
                                   subhalo_x_kpc[i],
                                   subhalo_y_kpc[i],
                                   subhalo_final_bound_masses[i],
                                   subhalo_infall_concentrations[i],
                                   z_lens, lens_cosmo)
        halo_list.append(halo)

    # combine the subhalos with line-of-sight halos
    subhalos_from_emulator = Realization.from_halos(halo_list, lens_cosmo, kwargs_halo_model={},
                                                    msheet_correction=False, rendering_classes=None)
    return cdm_halos_LOS.join(subhalos_from_emulator)

