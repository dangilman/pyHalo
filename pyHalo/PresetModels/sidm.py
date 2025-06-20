from pyHalo.PresetModels.cdm import CDM
from pyHalo.realization_extensions import RealizationExtensions


def SIDM_core_collapse(z_lens, z_source, mass_ranges_subhalos, mass_ranges_field_halos,
        probabilities_subhalos, probabilities_field_halos, sigma_sub=0.025, log10_sigma_sub=None, log10_dNdA=None,
        log_mlow=6., log_mhigh=10., concentration_model_subhalos='DIEMERJOYCE19', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='DIEMERJOYCE19', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_GALACTICUS', kwargs_truncation_model_subhalos={},
        infall_redshift_model='HYBRID_INFALL', kwargs_infall_model={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_m_host=13.3,  r_tidal=0.25,
        LOS_normalization=1.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None, host_scaling_factor=0.55,
        redshift_scaling_factor=0.37, two_halo_Lazar_correction=True, draw_poisson=True, c_host=6.0,
        add_globular_clusters=False, kwargs_globular_clusters=None, mass_threshold_sis=5*10**10,
        galaxy_model='GNFW', collapsed_halo_profile='TNFWC', **kwargs_halo_profile):

    """
    Generate a population of core-collapsed halos according the collapse probability in different mass bins

    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param mass_ranges_subhalos: a list of mass ranges for subhalo collapse, e.g.
    mass_ranges_subhalos = [[6, 8], [8, 9], ...]
    :param mass_ranges_field_halos: a list of mass ranges for subhalo collapse, e.g.
    mass_ranges_field_halos = [[6, 8], [8, 9], ...]
    :param probabilities_subhalos: a list of functions, or a list of the fraction of collapsed subhalos in each mass bin
    :param probabilities_field_halos: a list of functions, or a list of the fraction of collapsed field halos in each
    mass bin
    :param sigma_sub: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2]
    :param log10_sigma_sub: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2];
    this setting, as well as sigma_sub, will trigger the automatic scaling with host halo mass and redshift found by Gannon et al. (2025)
    - Note: this overrides sigma_sub, if provided
    :param log10_dNdA: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2], overrides sigma_sub
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
    :param infall_redshift_model: a string that specifies that infall redshift sampling distribution, see the LensCosmo
    class for details
    :param kwargs_infall_model: keyword arguments for the infall redshift model
    :param shmf_log_slope: the logarithmic slope of the subhalo mass function pivoting around 10^8 M_sun
    :param cone_opening_angle_arcsec: the opening angle of the rendering volume in arcsec
    :param log_m_host: log base 10 of the host halo mass [M_sun]
    :param r_tidal: the core radius of the host halo in units of the host halo scale radius. Subhalos are distributed
    in 3D with a cored NFW profile with this core radius
    :param LOS_normalization: rescales the amplitude of the line-of-sight mass function
    :param geometry_type: string that specifies the geometry of the rendering volume; options include
    DOUBLE_CONE, CONE, CYLINDER
    :param kwargs_cosmo: keyword arguments that specify the cosmology, see Cosmology/cosmology.py
    :param host_scaling_factor: the scaling with host halo mass of the projected number density of subhalos
    :param redshift_scaling_factor: the scaling with (1+z) of the projected number density of subhalos
    :param two_halo_Lazar_correction: bool; if True, adds the correction to the two-halo contribution from around the
    main deflector presented by Lazar et al. (2021)
    :param c_host: manually set host halo concentration
    :param add_globular_clusters: bool; include a population of globular clusters around image positions
    :param kwargs_globular_clusters: keyword arguments for the GC population; see documentation in RealizationExtensions
    :param mass_threshold_sis: the mass threshold above which NFW profiles become SIS/GNFW
    :param galaxy_model: the profile of massive line-of-sight galaxies; either SIS or GNFW
    :param collapsed_halo_profile: a string that specifies the collapsed halo profile, options include SPL_CORE, TNFWC, GNFW
    :param kwargs_halo: the keyword arguments for the collapsed halo profile
        - For SPL_CORE: should include 'log_slope_halo' and 'x_core_halo', the log profile slope and core size in units of
        NFW r_s
        - For GNFW: should include 'gamma_inner' and 'gamma_outer', the logarithmic profile slopes interior to and exterior
        to NFW r_s
        - For TNFWC: should include x_core_halo, the same parameter as in SPL_CORE (core size in units of rs)
        Note: if the truncation radius is smaller than rs, then rs will equal rt; if rt is smaller than rc, then rc
        will also equal rt (in this case, all three scales are equal in the profile drops like 1/r^5 outside the core)
    :return: a realization of collapsed SIDM halo profiles and NFW profiles
    """
    two_halo_contribution = True
    delta_power_law_index = 0.0
    subhalo_spatial_distribution = 'UNIFORM'
    cdm = CDM(z_lens, z_source, sigma_sub, log_mlow, log_mhigh, log10_sigma_sub, log10_dNdA,
              concentration_model_subhalos, kwargs_concentration_model_subhalos,
              concentration_model_fieldhalos, kwargs_concentration_model_fieldhalos,
              truncation_model_subhalos, kwargs_truncation_model_subhalos,
              truncation_model_fieldhalos, kwargs_truncation_model_fieldhalos,
              infall_redshift_model, kwargs_infall_model, subhalo_spatial_distribution,
              shmf_log_slope, cone_opening_angle_arcsec, log_m_host, r_tidal,
              LOS_normalization, two_halo_contribution, delta_power_law_index,
              geometry_type, kwargs_cosmo, host_scaling_factor,
                redshift_scaling_factor, two_halo_Lazar_correction, draw_poisson, c_host,
              add_globular_clusters, kwargs_globular_clusters, mass_threshold_sis, galaxy_model)
    extension = RealizationExtensions(cdm)
    index_collapsed = extension.core_collapse_by_mass(mass_ranges_subhalos, mass_ranges_field_halos,
        probabilities_subhalos, probabilities_field_halos)
    sidm = extension.add_core_collapsed_halos(index_collapsed, collapsed_halo_profile, **kwargs_halo_profile)
    return sidm

def SIDM_parametric(z_lens, z_source, log10_mass_ranges, log10_effective_cross_section_list, log10_subhalo_time_scaling=0.0,
        sigma_sub=0.025, log10_sigma_sub=None, log10_dNdA=None, evolving_SIDM_profile=True, x_core_halo=None,
        log_mlow=6., log_mhigh=10., concentration_model_subhalos='DIEMERJOYCE19', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='DIEMERJOYCE19', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_GALACTICUS', kwargs_truncation_model_subhalos={},
        infall_redshift_model='HYBRID_INFALL', kwargs_infall_model={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_m_host=13.3,  r_tidal=0.25,
        LOS_normalization=1.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None, add_globular_clusters=False,
        kwargs_globular_clusters=None):
    """

    Generates realizations of SIDM halos in terms of the effective scattering cross section in different mass bins using the
    parametric model by Yang et al.

    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param log10_mass_ranges: a list of lists specifying halo mass ranges, e.g. [[6, 8], [8, 10]]
    :param log10_effective_cross_section_list: a list of effective cross sections log10(sigma) for halos in each bin, e.g. [6.0, 1.0]
    :param log10_subhalo_time_scaling: a log10(number) that makes time pass quicker (>1) or lower (<1) for subhalos relative to
    field halos
    :param sigma_sub: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2]
    :param log10_sigma_sub: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2];
    this setting, as well as sigma_sub, will trigger the automatic scaling with host halo mass and redshift found by Gannon et al. (2025)
    - Note: this overrides sigma_sub, if provided
    :param log10_dNdA: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2], overrides sigma_sub
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
    :param infall_redshift_model: a string that specifies that infall redshift sampling distribution, see the LensCosmo
    class for details
    :param kwargs_infall_model: keyword arguments for the infall redshift model
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
    :param add_globular_clusters: bool; include a population of globular clusters around image positions
    :param kwargs_globular_clusters: keyword arguments for the GC population; see documentation in RealizationExtensions
    :return: a realization of dark matter structure in SIDM
    """
    two_halo_contribution = True
    delta_power_law_index = 0.0
    subhalo_spatial_distribution = 'UNIFORM'
    cdm = CDM(z_lens, z_source, sigma_sub, log_mlow, log_mhigh, log10_sigma_sub, log10_dNdA,
              concentration_model_subhalos, kwargs_concentration_model_subhalos,
              concentration_model_fieldhalos, kwargs_concentration_model_fieldhalos,
              truncation_model_subhalos, kwargs_truncation_model_subhalos,
              truncation_model_fieldhalos, kwargs_truncation_model_fieldhalos,
              infall_redshift_model, kwargs_infall_model, subhalo_spatial_distribution,
              shmf_log_slope, cone_opening_angle_arcsec, log_m_host, r_tidal,
              LOS_normalization, two_halo_contribution, delta_power_law_index,
              geometry_type, kwargs_cosmo)
    extension = RealizationExtensions(cdm)
    sidm = extension.toSIDM_from_cross_section(log10_mass_ranges,
                                                   log10_effective_cross_section_list,
                                                   log10_subhalo_time_scaling,
                                                    evolving_SIDM_profile,
                                                    x_core_halo)
    if add_globular_clusters:
        ext = RealizationExtensions(sidm)
        sidm = ext.add_globular_clusters(**kwargs_globular_clusters)
    return sidm

def SIDM_parametric_fixedbins(z_lens, z_source, log10_sigma_eff_mlow_8, log10_sigma_eff_8_mhigh,
        log10_subhalo_time_scaling=0.0, sigma_sub=0.025, log10_sigma_sub=None, log10_dNdA=None, log_mlow=6., log_mhigh=10.,
        concentration_model_subhalos='DIEMERJOYCE19', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='DIEMERJOYCE19', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_GALACTICUS', kwargs_truncation_model_subhalos={},
        infall_redshift_model='HYBRID_INFALL', kwargs_infall_model={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_m_host=13.3,  r_tidal=0.25,
        LOS_normalization=1.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None,
        add_globular_clusters=False, kwargs_globular_clusters=None, evolving_profile=True, x_core_halo=None):
    """
    This function uses the SIDM_parametric model with a preset list of logarithmic mass bins 10^log_mlow - 10^7.5,
    10^7.5-10^8.5, 10^8.5-10^10. The two parameters log10_sigma_eff_mlow_8 and log10_sigma_eff_8_mhigh set the cross section
     in each mass bin. For detailed documentation for the rest of the arguments, see the documentation in the SIDM_parametric function
    :param log10_sigma_eff_mlow_8: effective cross section in the mass range 10^log_mlow - 10^8
    :param log10_sigma_eff_8_mhigh: effective cross section in the mass range 10^8 - 10^log_mhigh
    :param log10_subhalo_time_scaling: a log10(number) that makes time pass quicker (>1) or lower (<1) for subhalos relative to
    field halos
    """
    if log_mlow > 8:
        raise Exception('to use this function log_mlow must be < 8')
    log10_mass_ranges = [[log_mlow, 8.0], [8.0, log_mhigh]]
    log10_effective_cross_section_list = [log10_sigma_eff_mlow_8, log10_sigma_eff_8_mhigh]
    return SIDM_parametric(
        z_lens, z_source, log10_mass_ranges, log10_effective_cross_section_list, log10_subhalo_time_scaling,
        sigma_sub, log10_sigma_sub, log10_dNdA, evolving_profile, x_core_halo,
        log_mlow, log_mhigh, concentration_model_subhalos, kwargs_concentration_model_subhalos,
        concentration_model_fieldhalos, kwargs_concentration_model_fieldhalos,
        truncation_model_subhalos, kwargs_truncation_model_subhalos,
        infall_redshift_model, kwargs_infall_model, truncation_model_fieldhalos,
        kwargs_truncation_model_fieldhalos, shmf_log_slope, cone_opening_angle_arcsec,
        log_m_host, r_tidal, LOS_normalization, geometry_type, kwargs_cosmo,
        add_globular_clusters,  kwargs_globular_clusters
    )
