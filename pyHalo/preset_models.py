"""
These functions define some preset halo mass function models that can be specified instead of individually specifying
a bunch of keyword arguments. This is intended for a quick and easy user interface. The definitions shown here can also
act as a how-to guide if one wants to explore more complicated descriptions of the halo mass function, as the models
presented here show what each keyword argument accepted by pyHalo does.
"""
from pyHalo.pyhalo import pyHalo
from pyHalo.realization_extensions import RealizationExtensions
from pyHalo.Cosmology.cosmology import Cosmology
import numpy as np
from pyHalo.utilities import de_broglie_wavelength

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
    elif name == 'SIDM':
        return SIDM
    elif name == 'ULDM':
        return ULDM
    else:
        raise Exception('preset model '+ str(name)+' not recognized!')

def CDM(z_lens, z_source, sigma_sub=0.025, shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_mlow=6.,
        log_mhigh=10., LOS_normalization=1., log_m_host=13.3, r_tidal='0.25Rs',
        mass_definition='TNFW', c0=None, log10c0=None,
        beta=None, zeta=None, two_halo_contribution=True, **kwargs_other):

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
    :param log_mlow: log10(minimum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param log_mhigh: log10(maximum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy
    :param kwargs_other: allows for additional keyword arguments to be specified when creating realization
    :param mass_definition: mass profile model for halos (TNFW is truncated NFW)

    The following optional keywords specify a concentration-mass relation parameterized as a power law in peak height.
    If they are not set in the function call, pyHalo assumes a default concentration-mass relation from Diemer&Joyce
    :param c0: amplitude of the mass-concentration relation at 10^8
    :param log10c0: logarithmic amplitude of the mass-concentration relation at 10^8 (only if c0_mcrelation is None)
    :param beta: logarithmic slope of the mass-concentration-relation pivoting around 10^8
    :param zeta: modifies the redshift evolution of the mass-concentration-relation
    :param two_halo_contribution: whether or not to include the two-halo term for correlated structure near
    the main deflector
    :return: a realization of CDM halos
    """

    kwargs_model_field = {'cone_opening_angle': cone_opening_angle_arcsec, 'mdef_los': mass_definition,
                          'mass_func_type': 'POWER_LAW', 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'LOS_normalization': LOS_normalization, 'log_m_host': log_m_host}

    kwargs_model_subhalos = {'cone_opening_angle': cone_opening_angle_arcsec, 'sigma_sub': sigma_sub,
                             'power_law_index': shmf_log_slope, 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                             'mdef_subs': mass_definition, 'mass_func_type': 'POWER_LAW', 'r_tidal': r_tidal}

    if any(x is not None for x in [c0, log10c0, beta, zeta]):

        if c0 is None:
            assert log10c0 is not None
            c0 = 10 ** log10c0

        assert beta is not None
        assert zeta is not None
        mc_model = {'custom': True,
                       'c0': c0, 'beta': beta, 'zeta': zeta}
        kwargs_mc_relation = {'mc_model': mc_model}
        kwargs_model_field.update(kwargs_mc_relation)
        kwargs_model_subhalos.update(kwargs_mc_relation)

    kwargs_model_field.update(kwargs_other)
    kwargs_model_subhalos.update(kwargs_other)

    # this will use the default cosmology. parameters can be found in defaults.py
    pyhalo = pyHalo(z_lens, z_source)
    # Using the render method will result a list of realizations
    realization_subs = pyhalo.render(['SUBHALOS'], kwargs_model_subhalos, nrealizations=1)[0]
    if two_halo_contribution:
        los_components = ['LINE_OF_SIGHT', 'TWO_HALO']
    else:
        los_components = ['LINE_OF_SIGHT']
    realization_line_of_sight = pyhalo.render(los_components, kwargs_model_field, nrealizations=1)[0]

    cdm_realization = realization_line_of_sight.join(realization_subs, join_rendering_classes=True)

    return cdm_realization

def WDM(z_lens, z_source, log_mc, log_mlow=6., log_mhigh=10., a_wdm_los=2.3, b_wdm_los=0.8, c_wdm_los=-1.,
                  a_wdm_sub=4.2, b_wdm_sub=2.5, c_wdm_sub=-0.2, cone_opening_angle_arcsec=6.,
                  sigma_sub=0.025, LOS_normalization=1., log_m_host= 13.3, power_law_index=-1.9, r_tidal='0.25Rs',
                    kwargs_suppression_mc_relation_field=None, suppression_model_field=None, kwargs_suppression_mc_relation_sub=None,
                  suppression_model_sub=None, two_halo_contribution=True, **kwargs_other):

    """

    This specifies the keywords for the Warm Dark Matter (WDM) halo mass function model presented by Lovell 2020
    (https://arxiv.org/pdf/2003.01125.pdf)

    The differential halo mass function is described by four parameters:
    1) log_mc - the log10 value of the half-mode mass, or the scale where the WDM mass function begins to deviate from CDM
    2) a_wdm - scale factor for the characteristic mass scale (see below)
    3) b_wdm - modifies the logarithmic slope of the WDM mass function (see below)
    4) c_wdm - modifies the logarithmic slope of the WDM mass function (see below)

    The defult parameterization for the mass function is:

    n_wdm / n_cdm = (1 + (a_wdm * m_c / m)^b_wdm) ^ c_wdm

    where m_c = 10**log_mc is the half-mode mass, and n_wdm and n_cdm are differential halo mass functions. Lovell 2020 find different fits
    to subhalos and field halos. For field halos, (a_wdm, b_wdm, c_wdm) = (2.3, 0.8, -1) while for subhalos
    (a_wdm_sub, b_wdm_sub, c_wdm_sub) = (4.2, 2.5, -0.2).

    WDM models also have reduced concentrations relative to CDM halos because WDM structure collapse later, when the Universe
    is less dense. The default suppresion to halo concentrations is implemented using the fitting function (Eqn. 17) presented by
    Bose et al. (2016) (https://arxiv.org/pdf/1507.01998.pdf), where the concentration relative to CDM is given by

    c_wdm / c_cdm = (1+z)^B(z) * (1 + c_scale * (m_c / m) ** c_power_inner) ^ c_power

    where m_c is the same as the definition for the halo mass function and (c_scale, c_power) = (60, -0.17). Note that the
    factor of 60 makes the effect on halo concentrations kick in on mass scales > m_c. This routine assumes the
    a mass-concentration for CDM halos given by Diemer & Joyce 2019 (https://arxiv.org/pdf/1809.07326.pdf)

    :param z_lens: the lens redshift
    :param z_source: the source redshift
    :param log_mc: log10(half mode mass) in units M_sun (no little h)
    :param log_mlow: log10(minimum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param log_mhigh: log10(maximum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param a_wdm_los: describes the line of sight WDM halo mass function (see above)
    :param b_wdm_los: describes the line of sight WDM halo mass function (see above)
    :param c_wdm_los: describes the line of sight WDM halo mass function (see above)
    :param a_wdm_sub: defines the WDM subhalo mass function (see above)
    :param b_wdm_sub: defines the WDM subhalo mass function (see above)
    :param c_wdm_sub: defines the WDM subhalo mass function (see above)
    :param cone_opening_angle: the opening angle in arcsec of the volume where halos are added
    :param sigma_sub: normalization of the subhalo mass function (see description in CDM preset model)
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param power_law_index: logarithmic slope of the subhalo mass function
    :param r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy

    ###################################################################################################
    The following keywords define how the WDM mass-concentration relation is suppressed relative to CDM

    :param kwargs_suppression_mc_relation_field: keyword arguments for the suppression function for field halo concentrations
    :param suppression_model_field: the type of suppression of the MC relation for field halos, either 'polynomial' or 'hyperbolic'. Default form is polynomial
    :param kwargs_suppression_mc_relation_sub: keyword arguments for the suppression function for subhalos
    :param suppression_model_sub: the type of suppression of the MC relation for subhalos, either 'polynomial' or 'hyperbolic'

    The form of the polynomial suppression function, f, is defined in terms of x = half-mode-mass / mass:

    f = (1 + c_scale * x ^ c_power_inner) ^ c_power

    The form of the hyperbolic suppression function, f, is (see functions in Halos.HaloModels.concentration)

    f = 1/2 [1 + tanh( (x - a_wdm)/2b_wdm ) ) ]
    ###################################################################################################

    :param kwargs_other: any other optional keyword arguments
    :param two_halo_contribution: whether or not to include the two-halo term for correlated structure near
    the main deflector
    :return: a realization of WDM halos
    """

    mass_definition = 'TNFW' # truncated NFW profile
    kwargs_model_field = {'a_wdm': a_wdm_los, 'b_wdm': b_wdm_los, 'c_wdm': c_wdm_los, 'log_mc': log_mc,
                          'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'cone_opening_angle': cone_opening_angle_arcsec, 'mdef_los': mass_definition,
                          'mass_func_type': 'POWER_LAW', 'LOS_normalization': LOS_normalization, 'log_m_host': log_m_host,
                          }

    if suppression_model_field is not None:
        kwargs_model_field['suppression_model'] = suppression_model_field
        kwargs_model_field['kwargs_suppression'] = kwargs_suppression_mc_relation_field

    kwargs_model_subhalos = {'a_wdm': a_wdm_sub, 'b_wdm': b_wdm_sub, 'c_wdm': c_wdm_sub, 'log_mc': log_mc,
                           'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'cone_opening_angle': cone_opening_angle_arcsec, 'sigma_sub': sigma_sub, 'mdef_subs': mass_definition,
                             'mass_func_type': 'POWER_LAW', 'power_law_index': power_law_index, 'r_tidal': r_tidal}


    if suppression_model_sub is not None:

        kwargs_model_subhalos['suppression_model'] = suppression_model_sub
        kwargs_model_subhalos['kwargs_suppression'] = kwargs_suppression_mc_relation_sub

    kwargs_model_field.update(kwargs_other)
    kwargs_model_subhalos.update(kwargs_other)

    # this will use the default cosmology. parameters can be found in defaults.py
    pyhalo = pyHalo(z_lens, z_source)
    # Using the render method will result a list of realizations
    realization_subs = pyhalo.render(['SUBHALOS'], kwargs_model_subhalos, nrealizations=1)[0]
    if two_halo_contribution:
        los_components = ['LINE_OF_SIGHT', 'TWO_HALO']
    else:
        los_components = ['LINE_OF_SIGHT']
    realization_line_of_sight = pyhalo.render(los_components, kwargs_model_field, nrealizations=1)[0]

    wdm_realization = realization_line_of_sight.join(realization_subs, join_rendering_classes=True)

    return wdm_realization


def SIDM(z_lens, z_source, cross_section_name, cross_section_class, kwargs_cross_section,
         kwargs_core_collapse_profile, deflection_angle_function, central_density_function, collapse_probability_function,
         t_sub=10, t_field=100, collapse_time_width=0.5, log_mlow=6., log_mhigh=10., cone_opening_angle_arcsec=6., sigma_sub=0.025,
         LOS_normalization=1., log_m_host=13.3, power_law_index=-1.9, r_tidal='0.25Rs', mdef='coreTNFW', mdef_collapse='SPL_CORE',
         realization_no_core_collapse=None, two_halo_contribution=True,
         **kwargs_other):

    """
    This generates realizations of self-interacting dark matter (SIDM) halos, inluding both cored and core-collapsed
    objects.

    :param z_lens: the lens redshift
    :param z_source: the source redshift
    :param cross_section_name: the name of the cross section implemented in SIDMpy
    :param cross_section_class: the class defined in SIDMpy for the SIDM cross section
    :param kwargs_cross_section: keyword arguments for the SIDM cross section class
    :param kwargs_core_collapse_profile: keyword arguments for the core collapse profile; possibilities include x_core_halo,
    x_match, and log_slope_halo, which are the core size in units of Rs, the radius (in units of Rs) where the core collapse
    profile encloses the same mass as the NFW profile, and the logarithmic slope of the core collapse profile
    (convention is for this to be a positive number)
    :param deflection_angle_function: a function that returns the deflection angles, to be passed into lenstronomy
    :param central_density_function: a function that computes the central density of cored halos
    :param collapse_probability_function: a function that computes the probability of core collapse for a given cross section
    :param velocity_dispersion_function: a function that computes the central velocity dispersion of an SIDM halo
    :param t_sub: core collapse timescale for subhalos
    :param t_field: core collapse timescale for field halos
    :param collapse_time_width: scatter in collapse times
    :param log_mlow: log10(minimum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param log_mhigh: log10(maximum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param cone_opening_angle: the opening angle in arcsec of the double cone geometry where halos are added
    :param sigma_sub: normalization of the subhalo mass function (see description in CDM preset model)
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param power_law_index: logarithmic slope of the subhalo mass function
    :param r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy
    :param kwargs_other: any addition keyword arguments
    :param mdef: the halo profile for halos that have not core collapsed
    :param mdef_collapse: the halo profile for halos that have core collapsed
    :param two_halo_contribution: whether or not to include the two-halo term for correlated structure near
    the main deflector
    :return: an instance of Realization that contains cored and core collapsed halos
    """

    kwargs_sidm =  {'cross_section_type': cross_section_name, 'kwargs_cross_section': kwargs_cross_section,
                       'SIDM_rhocentral_function': central_density_function,
                       'numerical_deflection_angle_class': deflection_angle_function}
    kwargs_sidm.update(kwargs_other)

    if realization_no_core_collapse is None:
        realization_no_core_collapse = CDM(z_lens, z_source, sigma_sub, power_law_index, cone_opening_angle_arcsec, log_mlow,
                              log_mhigh, LOS_normalization, log_m_host, r_tidal, mdef, **kwargs_sidm)

    ext = RealizationExtensions(realization_no_core_collapse)

    inds = ext.find_core_collapsed_halos(collapse_probability_function, cross_section_class,
                                  t_sub=t_sub, t_field=t_field, collapse_time_width=collapse_time_width)

    realization = ext.add_core_collapsed_halos(inds, mdef_collapse, **kwargs_core_collapse_profile)

    return realization

def ULDM(z_lens, z_source, log10_m_uldm, log10_fluc_amplitude=-0.8, fluctuation_size_scale=0.05,
         fluctuation_size_dispersion=0.2, n_fluc_scale=1.0, velocity_scale=200, log_mlow=6., log_mhigh=10., b_uldm=1.1, c_uldm=-2.2,
                  c_scale=21.42, c_power=-0.42, c_power_inner=1.62, cone_opening_angle_arcsec=6.,
                  sigma_sub=0.025, LOS_normalization=1., log_m_host= 13.3, power_law_index=-1.9, r_tidal='0.25Rs',
                  mass_definition='ULDM', uldm_plaw=1/3, scale_nfw=False, flucs=True,
                  flucs_shape='aperture', flucs_args={}, n_cut=50000, rescale_fluc_amp=True, r_ein=1.0, two_halo_contribution=True,
         **kwargs_other):

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

    :param z_lens: the lens redshift
    :param z_source: the source redshift
    :param log10_m_uldm: ULDM particle mass in log units, typically 1e-22 eV
    :param log10_fluc_amplitude: sets the amplitude of the fluctuations in the host dark matter halo.
    fluctuations are generated from a Guassian distriubtion with mean 0 and variance 10^log10_fluc_amplitude
    :param fluctuation_size_scale: half the size of an individual fluctuation; fluctuations are modeled as Gaussians with variance
    fluctuation_size_scale * lambda_dB, so their diameter is approximately 2 * fluctuation_size_scale * lambda_dB.
    To fit an overdenisty and an underdensiity inside one lambda_dB, you therefore need fluctuation_size_scale = 1/4
    :param fluctuation_size_dispersion: sets the variance of the distribution of fluctuation sizes, in units of fluctuation size:

    variance = fluctuation_size_dispersion * lambda_dB * fluctuation_size_scale

    To sumamrize, individual fluctuations are modeled as Gaussians with a mean fluctuation_size_scale * lambda_dB and a
    variance fluctuation_size_dispersion * lambda_dB
    :param n_fluc_scale: rescales the total number of fluctuations
    :param velocity_scale: velocity for de Broglie wavelength calculation in km/s
    :param log_mhigh: log10(maximum halo mass) rendered (mass definition is M200 w.r.t. critical density)
    :param b_uldm: defines the ULDM mass function (see above)
    :param c_uldm: defines the ULDM mass function (see above)
    :param c_scale: scale where concentrations in ULDM deviate from CDM (see WDMLovell2020)
    :param c_power: modification to logarithmic slope of mass-concentration relation (see WDMLovell2020)
    :param cone_opening_angle: the opening angle in arcsec of the double cone geometry where halos are added
    :param sigma_sub: normalization of the subhalo mass function (see description in CDM preset model)
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param power_law_index: logarithmic slope of the subhalo mass function
    :param r_tidal: subhalos are distributed following a cored NFW profile with a core radius r_tidal. This is intended
    to account for tidal stripping of halos that pass close to the central galaxy
    :param mass_definition: mass profile model for halos
    :param uldm_plaw: ULDM core radius-halo mass power law exponent, typically 1/3
    :param scale_nfw: boolean specifiying whether or not to scale the NFW component (can improve mass accuracy)
    :param flucs: Boolean specifying whether or not to include density fluctuations in the main deflector halo
    :param flucs_shape: String specifying how to place fluctuations, see docs in realization_extensions.add_ULDM_fluctuations
    :param fluc_args: Keyword arguments for specifying the fluctuations, see docs in realization_extensions.add_ULDM_fluctuations
    :param rescale_fluc_amp: Boolean specifying whether re-scale fluctuation amplitudes by sqrt(n_cut / n), where n
    is the total number of fluctuations in the given area and n_cut (defined below) is the maximum number to generate
    :param einstein_radius: Einstein radius of main deflector halo in kpc
    :param n_cut: Number of fluctuations above which to start cancelling
    :param r_ein: the Einstein radius in arcseconds
    :param kwargs_other: any other optional keyword arguments
    :param two_halo_contribution: whether or not to include the two-halo term for correlated structure near
    the main deflector
    :return: a realization of ULDM halos
    """
    # constants
    m22 = 10**(log10_m_uldm + 22)
    log_m0 = np.log10(1.6e10 * m22**(-4/3))
    M_min0 = 4.4e7 * m22**(-3/2) # M_solar

    a_uldm = 1 # set to unity since Schive et al. 2016 do not have an analogous parameter

    #compute M_min as described in documentation
    a = lambda z: (1+z)**(-1)
    O_m = lambda z: Cosmology().astropy.Om(z)
    zeta = lambda z: (18*np.pi**2 + 82*(O_m(z)-1) - 39*(O_m(z)-1)**2) / O_m(z)
    m_min = lambda z: a(z)**(-3/4) * (zeta(z)/zeta(0))**(1/4) * M_min0
    log_m_min = lambda z: np.log10(m_min(z))

    if log_m_min(z_lens) >= log_mlow:
        log_mlow = log_m_min(z_lens) # only use M_min for minimum halo mass if it is above input 'log_mlow'
    kwargs_model_field = {'a_wdm': a_uldm, 'b_wdm': b_uldm, 'c_wdm': c_uldm, 'log_mc': log_m0,
                          'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'cone_opening_angle': cone_opening_angle_arcsec, 'mdef_los': mass_definition,
                          'mass_func_type': 'POWER_LAW', 'LOS_normalization': LOS_normalization, 'log_m_host': log_m_host}

    kwargs_model_subhalos = {'a_wdm': a_uldm, 'b_wdm': b_uldm, 'c_wdm': c_uldm, 'log_mc': log_m0,
                          'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                          'cone_opening_angle': cone_opening_angle_arcsec, 'sigma_sub': sigma_sub, 'mdef_subs': mass_definition,
                             'mass_func_type': 'POWER_LAW', 'power_law_index': power_law_index, 'r_tidal': r_tidal}

    kwargs_model_subhalos.update({'suppression_model': 'polynomial'})
    kwargs_model_field.update({'suppression_model': 'polynomial'})

    kwargs_suppression_mcrelation = {'c_scale': c_scale,
                                     'c_power': c_power,
                                     'c_power_inner': c_power_inner,
                                     'mc_suppression_redshift_evolution': False}

    kwargs_model_subhalos.update({'kwargs_suppression': kwargs_suppression_mcrelation})
    kwargs_model_field.update({'kwargs_suppression': kwargs_suppression_mcrelation})

    kwargs_uldm = {'log10_m_uldm': log10_m_uldm, 'uldm_plaw': uldm_plaw, 'scale_nfw':scale_nfw}

    kwargs_model_field.update(kwargs_uldm)
    kwargs_model_subhalos.update(kwargs_uldm)

    kwargs_model_field.update(kwargs_other)
    kwargs_model_subhalos.update(kwargs_other)

    # this will use the default cosmology. parameters can be found in defaults.py
    pyhalo = pyHalo(z_lens, z_source)
    # Using the render method will result a list of realizations
    realization_subs = pyhalo.render(['SUBHALOS'], kwargs_model_subhalos, nrealizations=1)[0]
    if two_halo_contribution:
        los_components = ['LINE_OF_SIGHT', 'TWO_HALO']
    else:
        los_components = ['LINE_OF_SIGHT']
    realization_line_of_sight = pyhalo.render(los_components, kwargs_model_field, nrealizations=1)[0]
    uldm_realization = realization_line_of_sight.join(realization_subs, join_rendering_classes=True)

    if flucs: # add fluctuations to realization
        ext = RealizationExtensions(uldm_realization)
        lambda_dB = de_broglie_wavelength(log10_m_uldm, velocity_scale) # de Broglie wavelength in kpc

        if flucs_args=={}:
            raise Exception('Must specify fluctuation arguments, see realization_extensions.add_ULDM_fluctuations')

        a_fluc = 10 ** log10_fluc_amplitude
        m_psi = 10 ** log10_m_uldm

        zlens_ref, zsource_ref = 0.5, 2.0
        mhost_ref = 10**13.3
        rein_ref = 1.0
        r_perp_ref = rein_ref * uldm_realization.lens_cosmo.cosmo.kpc_proper_per_asec(zlens_ref)

        sigma_crit_ref = uldm_realization.lens_cosmo.get_sigma_crit_lensing(zlens_ref, zsource_ref)
        c_host_ref = uldm_realization.lens_cosmo.NFW_concentration(mhost_ref, z_lens, scatter=False)
        rhos_ref, rs_ref, _ = uldm_realization.lens_cosmo.NFW_params_physical(mhost_ref, c_host_ref, zlens_ref)
        xref = r_perp_ref/rs_ref
        if xref < 1:
            Fxref = np.arctanh(np.sqrt(1 - xref ** 2)) / np.sqrt(1 - xref ** 2)
        else:
            Fxref = np.arctan(np.sqrt(-1 + xref ** 2)) / np.sqrt(-1 + xref ** 2)
        sigma_host_ref = 2 * rhos_ref * rs_ref * (1-Fxref)/(xref**2 - 1)

        r_perp = r_ein * uldm_realization.lens_cosmo.cosmo.kpc_proper_per_asec(z_lens)
        sigma_crit = uldm_realization.lens_cosmo.get_sigma_crit_lensing(z_lens, z_source)
        c_host = uldm_realization.lens_cosmo.NFW_concentration(10**log_m_host, z_lens, scatter=True)
        rhos, rs, _ = uldm_realization.lens_cosmo.NFW_params_physical(10**log_m_host, c_host, z_lens)
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
                                fluctuation_size_variance=lambda_dB * fluctuation_size_scale *
                                                          fluctuation_size_dispersion,
                                n_fluc_scale=n_fluc_scale,
                                shape=flucs_shape,
                                args=flucs_args,
                                n_cut=n_cut, rescale_fluc_amp=rescale_fluc_amp)

    return uldm_realization
