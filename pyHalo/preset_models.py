"""
These functions define some preset halo mass function models that can be specified instead of individually specifying
a bunch of keyword arguments. This is intended for a quick and easy user interface. The definitions shown here can also
act as a how-to guide if one wants to explore more complicated descriptions of the halo mass function, as the models
presented here show what each keyword argument accepted by pyHalo does.
"""
from pyHalo.pyhalo import pyHalo


def CDM(z_lens, z_source, sigma_sub=0.025, shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_mlow=6,
        log_mhigh=10, kwargs_model_other={}):

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
    :param kwargs_model_other: any other keyword arguments one wants to pass to realization
    :return:
    """

    mass_definition = 'TNFW'  # truncated NFW profile
    kwargs_model_field = {'cone_opening_angle': cone_opening_angle_arcsec, 'mdef_los': mass_definition,
                          'mass_func_type': 'POWER_LAW', 'log_mlow': log_mlow, 'log_mhigh': log_mhigh}

    kwargs_model_subhalos = {'cone_opening_angle': cone_opening_angle_arcsec, 'sigma_sub': sigma_sub,
                             'power_law_index': shmf_log_slope, 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                             'mdef_main': mass_definition, 'mass_func_type': 'POWER_LAW'}

    kwargs_model_field.update(kwargs_model_other)
    kwargs_model_subhalos.update(kwargs_model_other)

    """
    Optional parameters for line of sight:

    LOS_normalization: the amplitude of the line of sight mass function relative to Sheth-Tormen
    Note: you can optionally turn off the line of sight contribution by specifying LOS_normalization=0 in kwargs_model_other

    log_m_host: host halo mass in M_sun (see CDM preset model for details). The line of sight mass function
    needs the host halo mass because the amount of correlated structure around the host depends on the host halo mass.
    """

    optional_params_field = {'LOS_normalization': 1., 'log_m_host': 13.3}
    for param in optional_params_field.keys():
        if param not in kwargs_model_field.keys():
            kwargs_model_field[param] = optional_params_field[param]

    """
    Optional parameters for subhalo mass function:

    log_m_host: host halo mass in M_sun (see CDM preset model for details).

    r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy
    """

    optional_params_subhalos = {'log_m_host': 13.3, 'r_tidal': '0.25Rs'}
    for param in optional_params_subhalos.keys():
        if param not in kwargs_model_subhalos.keys():
            kwargs_model_subhalos[param] = optional_params_subhalos[param]

    # this will use the default cosmology. parameters can be found in defaults.py
    pyhalo = pyHalo(z_lens, z_source)
    # Using the render method will result a list of realizations
    realization_subs = pyhalo.render('main_lens', kwargs_model_subhalos, nrealizations=1)[0]
    realization_line_of_sight = pyhalo.render('line_of_sight', kwargs_model_field, nrealizations=1)[0]

    cdm_realization = realization_line_of_sight.join(realization_subs, join_rendering_classes=True)

    return cdm_realization

def WDMLovell2020(z_lens, z_source, log_mc, log_mlow=6., log_mhigh=10., a_wdm_los=2.3, b_wdm_los=0.8, c_wdm_los=-1., a_wdm_sub=4.2, b_wdm_sub=2.5,
                   c_wdm_sub=-0.2, c_scale=60., c_power=-0.17, cone_opening_angle_arcsec=6., sigma_sub=0.025, kwargs_model_other={}):

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
    :param cone_opening_angle_arcsec: the opening angle in arcsec of the double cone geometry where halos are added
    :param sigma_sub: normalization of the subhalo mass function (see description in CDM preset model)
    :param kwargs_model_other: (optional) additional keyword arguments for the model
    :return: a realization of WDM halos

    :return:
    """
    mass_definition = 'TNFW' # truncated NFW profile
    kwargs_model_field = {'a_wdm': a_wdm_los, 'b_wdm': b_wdm_los, 'c_wdm': c_wdm_los, 'log_mc': log_mc,
                          'c_scale': c_scale, 'c_power': c_power, 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'cone_opening_angle': cone_opening_angle_arcsec, 'mdef_los': mass_definition,
                          'mass_func_type': 'POWER_LAW'}

    kwargs_model_subhalos = {'a_wdm': a_wdm_sub, 'b_wdm': b_wdm_sub, 'c_wdm': c_wdm_sub, 'log_mc': log_mc,
                          'c_scale': c_scale, 'c_power': c_power, 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'cone_opening_angle': cone_opening_angle_arcsec, 'sigma_sub': sigma_sub, 'mdef_main': mass_definition,
                             'mass_func_type': 'POWER_LAW'}

    kwargs_model_field.update(kwargs_model_other)
    kwargs_model_subhalos.update(kwargs_model_other)

    """
    LOS_normalization: the amplitude of the line of sight mass function relative to Sheth-Tormen
    log_m_host: host halo mass in M_sun (see CDM preset model for details). The line of sight mass function
    needs the host halo mass because the amount of correlated structure around the host depends on the host halo mass.
    """
    optional_params_field = {'LOS_normalization': 1., 'log_m_host': 13.3}
    for param in optional_params_field.keys():
        if param not in kwargs_model_field.keys():
            kwargs_model_field[param] = optional_params_field[param]

    """
    power_law_index: the logarithmic slope of the differential subhalo mass function
    log_m_host: host halo mass in M_sun (see CDM preset model for details).
    r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy
    """

    optional_params_subhalos = {'power_law_index': -1.9, 'log_m_host': 13.3, 'r_tidal': '0.25Rs'}
    for param in optional_params_subhalos.keys():
        if param not in kwargs_model_subhalos.keys():
            kwargs_model_subhalos[param] = optional_params_subhalos[param]

    # this will use the default cosmology. parameters can be found in defaults.py
    pyhalo = pyHalo(z_lens, z_source)
    # Using the render method will result a list of realizations
    realization_subs = pyhalo.render('main_lens', kwargs_model_subhalos, nrealizations=1)[0]
    realization_line_of_sight = pyhalo.render('line_of_sight', kwargs_model_field, nrealizations=1)[0]

    wdm_realization = realization_line_of_sight.join(realization_subs, join_rendering_classes=True)

    return wdm_realization

