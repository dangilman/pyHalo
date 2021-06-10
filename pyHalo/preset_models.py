"""
These functions define some preset halo mass function models that can be specified instead of individually specifying
a bunch of keyword arguments. This is intended for a quick and easy user interface. The definitions shown here can also
act as a how-to guide if one wants to explore more complicated descriptions of the halo mass function, as the models
presented here show what each keyword argument accepted by pyHalo does.
"""
from pyHalo.pyhalo import pyHalo
from pyHalo.realization_extensions import RealizationExtensions

def CDM(z_lens, z_source, sigma_sub=0.025, shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_mlow=6.,
        log_mhigh=10., LOS_normalization=1., log_m_host=13.3, r_tidal='0.25Rs', mass_definition='TNFW', **kwargs_other):

    """

    This specifies the keywords for a CDM halo mass function model with a subhalo mass function described by a power law
    and a line of sight halo mass function described by Sheth-Tormen.

    The subhalo mass function is parameterized as
    d^N / dmdA = shmf_norm / m0 * (m/m0)^power_law_index * F(M_host, z)

    with a pivot mass m0=10^8. In this parameterization, shmf_norm has units of 1/area, or kpc^-2. CDM prediction is
    something like 0.01 - 0.05, but this depends on the efficiency of stripping and what objects around the host
    halo you consider a subhalos (are splashback halos subhalos?).

    The function F(M_host, z) factors the evolution of the projected number density of subhalos with host halo mass
    and redshift out of the projected number density, so sigma_sub should be common to each host halo. The function
    F(M_host, z) is calibrated with the semi-analytic modeling tool galacticus (https://arxiv.org/pdf/1008.1786.pdf):

    F(M_host, z) = a * log10(M_host / 10^13) + b * (1+z)
    with a = 0.88 and b = 1.7.

    :param z_lens: main deflector redshift
    :param z_source: sourcee redshift
    :param sigma_sub: normalization of the subhalo mass function
    :param shmf_log_slope: logarithmic slope of the subhalo mass function
    :param cone_opening_angle_arcsec: the opening angle of the double cone rendering volume in arcsec
    :param log_mlow: log10(minimum halo mass) rendered
    :param log_mhigh: log10(maximum halo mass) rendered (mass definition is M200 w.r.t. critical density
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy
    :param kwargs_other: allows for additional keyword arguments to be specified when creating realization
    :param mass_definition: mass profile model for halos (TNFW is truncated NFW)
    :return: a realization of CDM halos
    """

    kwargs_model_field = {'cone_opening_angle': cone_opening_angle_arcsec, 'mdef_los': mass_definition,
                          'mass_func_type': 'POWER_LAW', 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'LOS_normalization': LOS_normalization, 'log_m_host': log_m_host}

    kwargs_model_subhalos = {'cone_opening_angle': cone_opening_angle_arcsec, 'sigma_sub': sigma_sub,
                             'power_law_index': shmf_log_slope, 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                             'mdef_subs': mass_definition, 'mass_func_type': 'POWER_LAW', 'r_tidal': r_tidal}

    kwargs_model_field.update(kwargs_other)
    kwargs_model_subhalos.update(kwargs_other)

    # this will use the default cosmology. parameters can be found in defaults.py
    pyhalo = pyHalo(z_lens, z_source)
    # Using the render method will result a list of realizations

    realization_subs = pyhalo.render(['SUBHALOS'], kwargs_model_subhalos, nrealizations=1)[0]
    realization_line_of_sight = pyhalo.render(['LINE_OF_SIGHT', 'TWO_HALO'], kwargs_model_field, nrealizations=1)[0]

    cdm_realization = realization_line_of_sight.join(realization_subs, join_rendering_classes=True)

    return cdm_realization

def WDMLovell2020(z_lens, z_source, log_mc, log_mlow=6., log_mhigh=10., a_wdm_los=2.3, b_wdm_los=0.8, c_wdm_los=-1.,
                  a_wdm_sub=4.2, b_wdm_sub=2.5, c_wdm_sub=-0.2, c_scale=60., c_power=-0.17, cone_opening_angle_arcsec=6.,
                  sigma_sub=0.025, LOS_normalization=1., log_m_host= 13.3, power_law_index=-1.9, r_tidal='0.25Rs',
                  **kwargs_other):

    """

    This specifies the keywords for the Warm Dark Matter (WDM) halo mass function model presented by Lovell 2020
    (https://arxiv.org/pdf/2003.01125.pdf)

    The differential halo mass function is described by four parameters:
    1) log_mc - the log10 value of the half-mode mass, or the scale where the WDM mass function begins to deviate from CDM
    2) a_wdm - scale factor for the characteristic mass scale (see below)
    3) b_wdm - modifies the logarithmic slope of the WDM mass function (see below)
    4) c_wdm - modifies the logarithmic slope of the WDM mass function (see below)

    The parameterization for the mass function is:

    n_wdm / n_cdm = (1 + (a_wdm * m_c / m)^b_wdm) ^ c_wdm

    where m_c = 10**log_mc and n_wdm and n_cdm are differential halo mass functions. Lovell 2020 find different fits
    to subhalos and field halos. For field halos, (a_wdm, b_wdm, c_wdm) = (2.3, 0.8, -1) while for subhalos
    (a_wdm, b_wdm, c_wdm) = (4.2, 2.5, -0.2).

    WDM models also have reduced concentrations relative to CDM halos because WDM structure collapse later when the Universe
    is less dense. The suppresion to halo concentrations is implemented using the fitting function (Eqn. 17) presented by
    Bose et al. (2016) (https://arxiv.org/pdf/1507.01998.pdf), where the concentration relative to CDM is given by

    c_wdm / c_cdm = (1+z)^B(z) * (1 + c_scale * m_c / m) ^ c_power

    where m_c is the same as the definition for the halo mass function and (c_scale, c_power) = (60, -0.17). Note that the
    factor of 60 makes the effect on halo concentrations kick in on mass scales > m_c. This routine assumes the
    a mass-concentration for CDM halos given by Diemer & Joyce 2019 (https://arxiv.org/pdf/1809.07326.pdf)

    :param z_lens: the lens redshift
    :param z_source: the source redshift
    :param log_mc: log10(half mode mass) in units M_sun (no little h)
    :param log_mlow: log10(minimum halo mass) rendered
    :param log_mhigh: log10(maximum halo mass) rendered (mass definition is M200 w.r.t. critical density
    :param a_wdm_los: describes the line of sight WDM halo mass function (see above)
    :param b_wdm_los: describes the line of sight WDM halo mass function (see above)
    :param c_wdm_los: describes the line of sight WDM halo mass function (see above)
    :param a_wdm_sub: defines the WDM subhalo mass function (see above)
    :param b_wdm_sub: defines the WDM subhalo mass function (see above)
    :param c_wdm_sub: defines the WDM subhalo mass function (see above)
    :param c_scale: scale where concentrations in WDM deviate from CDM
    :param c_power: modification to logarithmic slope of mass-concentration relation
    :param cone_opening_angle: the opening angle in arcsec of the double cone geometry where halos are added
    :param sigma_sub: normalization of the subhalo mass function (see description in CDM preset model)
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param power_law_index: logarithmic slope of the subhalo mass function
    :param r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy
    :param kwargs_other: any other optional keyword arguments

    :return: a realization of WDM halos
    """
    mass_definition = 'TNFW' # truncated NFW profile
    kwargs_model_field = {'a_wdm': a_wdm_los, 'b_wdm': b_wdm_los, 'c_wdm': c_wdm_los, 'log_mc': log_mc,
                          'c_scale': c_scale, 'c_power': c_power, 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'cone_opening_angle': cone_opening_angle_arcsec, 'mdef_los': mass_definition,
                          'mass_func_type': 'POWER_LAW', 'LOS_normalization': LOS_normalization, 'log_m_host': log_m_host,
                          }

    kwargs_model_subhalos = {'a_wdm': a_wdm_sub, 'b_wdm': b_wdm_sub, 'c_wdm': c_wdm_sub, 'log_mc': log_mc,
                          'c_scale': c_scale, 'c_power': c_power, 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'cone_opening_angle': cone_opening_angle_arcsec, 'sigma_sub': sigma_sub, 'mdef_subs': mass_definition,
                             'mass_func_type': 'POWER_LAW', 'power_law_index': power_law_index, 'r_tidal': r_tidal}

    kwargs_model_field.update(kwargs_other)
    kwargs_model_subhalos.update(kwargs_other)

    # this will use the default cosmology. parameters can be found in defaults.py
    pyhalo = pyHalo(z_lens, z_source)
    # Using the render method will result a list of realizations
    realization_subs = pyhalo.render(['SUBHALOS'], kwargs_model_subhalos, nrealizations=1)[0]
    realization_line_of_sight = pyhalo.render(['LINE_OF_SIGHT', 'TWO_HALO'], kwargs_model_field, nrealizations=1)[0]

    wdm_realization = realization_line_of_sight.join(realization_subs, join_rendering_classes=True)

    return wdm_realization


def WDMGeneral(z_lens, z_source, log_mc, log_mlow=6., log_mhigh=10., a_wdm=2.3, b_wdm=0.8, c_wdm=-1.,
               c_scale=60., c_power=-0.17, cone_opening_angle_arcsec=6., sigma_sub=0.025, LOS_normalization=1.,
               log_m_host=13.3, power_law_index=-1.9, r_tidal='0.25Rs', **kwargs_other):

    """
    This is a more restricted form for the warm dark matter halo mass function than WDMLovell2020 because
    the subhalo mass function and field halo mass function share the same functional form
    :param z_lens: the lens redshift
    :param z_source: the source redshift
    :param log_mc: log10(half mode mass) in units M_sun (no little h)
    :param log_mlow: log10(minimum halo mass) rendered
    :param log_mhigh: log10(maximum halo mass) rendered (mass definition is M200 w.r.t. critical density
    :param a_wdm: describes the line of sight WDM halo mass function (see documention in WDMLovell2020)
    :param b_wdm: describes the line of sight WDM halo mass function (see documention in WDMLovell2020)
    :param c_wdm: describes the line of sight WDM halo mass function (see documention in WDMLovell2020)
    :param c_scale: scale where concentrations in WDM deviate from CDM
    :param c_power: modification to logarithmic slope of mass-concentration relation
    :param cone_opening_angle: the opening angle in arcsec of the double cone geometry where halos are added
    :param sigma_sub: normalization of the subhalo mass function (see description in CDM preset model)
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param power_law_index: logarithmic slope of the subhalo mass function
    :param r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy
    :param kwargs_other: any other optional keyword arguments

    :return: a realization of WDM halos
    """
    return WDMLovell2020(z_lens, z_source, log_mc, log_mlow, log_mhigh, a_wdm, b_wdm, c_wdm,
                         a_wdm, b_wdm, c_wdm, c_scale, c_power, cone_opening_angle_arcsec, sigma_sub, LOS_normalization,
                         log_m_host, power_law_index, r_tidal, **kwargs_other)

def SIDM(z_lens, z_source, cross_section_name, cross_section_class, kwargs_cross_section,
         logarithmic_core_collapse_slope, core_collapsed_core_size,
         deflection_angle_function, central_density_function, evolution_timescale_function,
         velocity_dispersion_function, t_sub=10, t_field=100, log_mlow=6., log_mhigh=10., cone_opening_angle_arcsec=6., sigma_sub=0.025,
         LOS_normalization=1., log_m_host=13.3, power_law_index=-1.9, r_tidal='0.25Rs', **kwargs_other):

    """
    This generates realizations of self-interacting dark matter (SIDM) halos, inluding both cored and core-collapsed
    objects.

    :param z_lens: the lens redshift
    :param z_source: the source redshift
    :param cross_section_name: the name of the cross section implemented in SIDMpy
    :param cross_section_class: the class defined in SIDMpy for the SIDM cross section
    :param kwargs_cross_section: keyword arguments for the SIDM cross section class
    :param logarithmic_core_collapse_slope: logarithmic slope of core collapsed halos
    :param core_collapsed_core_size: central core size of core collapsed halos (in units of Rs)
    :param deflection_angle_function: a function that returns the deflection angles, to be passed into lenstronomy
    :param central_density_function: a function that computes the central density of cored halos
    :param evolution_timescale_function: a function that computes the evolution timescale t_0 for a given cross section
    (for example, Equation 9 in Gilman et al. 2021)
    :param velocity_dispersion_function: a function that computes the central velocity dispersion of an SIDM halo
    :param t_sub: core collapse timescale for subhalos
    :param t_field: core collapse timescale for field halos
    :param log_mlow: log10(minimum halo mass) rendered
    :param log_mhigh: log10(maximum halo mass) rendered (mass definition is M200 w.r.t. critical density
    :param cone_opening_angle: the opening angle in arcsec of the double cone geometry where halos are added
    :param sigma_sub: normalization of the subhalo mass function (see description in CDM preset model)
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param power_law_index: logarithmic slope of the subhalo mass function
    :param r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy
    :param kwargs_other: any addition keyword arguments
    :return: an instance of Realization that contains cored and core collapsed halos
    """

    kwargs_sidm =  {'cross_section_type': cross_section_name, 'kwargs_cross_section': kwargs_cross_section,
                       'SIDM_rhocentral_function': central_density_function,
                       'numerical_deflection_angle_class': deflection_angle_function}
    kwargs_sidm.update(kwargs_other)

    realization_no_core_collapse = CDM(z_lens, z_source, sigma_sub, power_law_index, cone_opening_angle_arcsec, log_mlow,
                              log_mhigh, LOS_normalization, log_m_host, r_tidal, 'coreTNFW', **kwargs_sidm)

    ext = RealizationExtensions(realization_no_core_collapse)

    inds = ext.find_core_collapsed_halos(evolution_timescale_function, velocity_dispersion_function,
                                         cross_section_class, t_sub=t_sub, t_field=t_field)

    realization = ext.add_core_collapsed_halos(inds, log_slope_halo=logarithmic_core_collapse_slope,
                                                       x_core_halo=core_collapsed_core_size)

    return realization


