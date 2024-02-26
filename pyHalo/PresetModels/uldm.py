import numpy as np

from pyHalo.PresetModels.wdm import WDM
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.pyhalo import pyHalo
from pyHalo.realization_extensions import RealizationExtensions
from pyHalo.utilities import MinHaloMassULDM, de_broglie_wavelength


def ULDM(z_lens, z_source, log10_m_uldm, log10_fluc_amplitude=-0.8, fluctuation_size_scale=0.05,
          fluctuation_size_dispersion=0.2, n_fluc_scale=1.0, velocity_scale=200, sigma_sub=0.025,
         log10_sigma_sub=None, log_mlow=6., log_mhigh=10.,
        mass_function_model_subhalos='SHMF_SCHIVE2016', kwargs_mass_function_subhalos={},
        mass_function_model_fieldhalos='SCHIVE2016', kwargs_mass_function_fieldhalos={},
        concentration_model_subhalos='LAROCHE2022', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='LAROCHE2022', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_GALACTICUS', kwargs_truncation_model_subhalos={},
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
    :param log10_sigma_sub: optional setting of sigma_sub in log10-scale (useful for log-uniform priors); if this is specified
    it overwrites the value of sigma_sub
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
                  'log10_sigma_sub': log10_sigma_sub,
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
