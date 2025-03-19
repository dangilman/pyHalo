from copy import copy
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw
from pyHalo.Rendering.SpatialDistributions.nfw import ProjectedNFW
from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform, Uniform
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.pyhalo import pyHalo
from pyHalo.truncation_models import truncation_models
from pyHalo.realization_extensions import RealizationExtensions


def CDM(z_lens, z_source, sigma_sub=0.025, log_mlow=6., log_mhigh=10., log10_sigma_sub=None,
        concentration_model_subhalos='LUDLOW2016', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='LUDLOW2016', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_GALACTICUS', kwargs_truncation_model_subhalos={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        infall_redshift_model='HYBRID_INFALL', kwargs_infall_model={},
        subhalo_spatial_distribution='UNIFORM',
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6.,
        log_m_host=13.3,  r_tidal=0.25, LOS_normalization=1.0, two_halo_contribution=True,
        delta_power_law_index=0.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None, host_scaling_factor=0.55,
        redshift_scaling_factor=0.37, two_halo_Lazar_correction=True, draw_poisson=True, c_host=6.0,
        add_globular_clusters=False, kwargs_globular_clusters=None, mass_threshold_sis=5*10**10, galaxy_model='GNFW'):
    """
    This class generates realizations of dark matter structure in Cold Dark Matter

    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param sigma_sub: amplitude of the subhalo mass function at 10^8 solar masses in units [# of halos / kpc^2]
    :param log10_sigma_sub: optional setting of sigma_sub in log10-scale (useful for log-uniform priors); if this is specified
    it overwrites the value of sigma_sub
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
    :param subhalo_spatial_distribution: the spatial distribution model for subhalos
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
    :param host_scaling_factor: the scaling with host halo mass of the projected number density of subhalos
    :param redshift_scaling_factor: the scaling with (1+z) of the projected number density of subhalos
    :param two_halo_Lazar_correction: bool; if True, adds the correction to the two-halo contribution from around the
    main deflector presented by Lazar et al. (2021)
    :param c_host: manually set host halo concentration
    :param add_globular_clusters: bool; include a population of globular clusters around image positions
    :param kwargs_globular_clusters: keyword arguments for the GC population; see documentation in RealizationExtensions
    :param mass_threshold_sis: the mass threshold above which NFW profiles become SIS/GNFW
    :param galaxy_model: the profile of massive line-of-sight galaxies; either SIS or GNFW
    :return: a realization of dark matter halos
    """
    # FIRST WE CREATE AN INSTANCE OF PYHALO, WHICH SETS THE COSMOLOGY
    pyhalo = pyHalo(z_lens, z_source, kwargs_cosmo)
    # WE ALSO SPECIFY THE GEOMETRY OF THE RENDERING VOLUME
    geometry = Geometry(pyhalo.cosmology, z_lens, z_source,
                        cone_opening_angle_arcsec, geometry_type)
    if infall_redshift_model == 'HYBRID_INFALL':
        kwargs_infall_model['log_m_host'] = log_m_host
    pyhalo.lens_cosmo.setup_infall_model(infall_redshift_model, kwargs_infall_model)
    # NOW WE SET THE MASS FUNCTION CLASSES FOR SUBHALOS AND FIELD HALOS
    # NOTE: MASS FUNCTION CLASSES SHOULD NOT BE INSTANTIATED HERE
    mass_function_model_subhalos = CDMPowerLaw
    mass_function_model_fieldhalos = ShethTormen

    # set the density profile definition
    mdef_subhalos = 'TNFW'
    mdef_field_halos = 'TNFW'

    kwargs_concentration_model_subhalos['cosmo'] = pyhalo.astropy_cosmo
    kwargs_concentration_model_fieldhalos['cosmo'] = pyhalo.astropy_cosmo

    # SET THE CONCENTRATION-MASS RELATION FOR SUBHALOS AND FIELD HALOS
    model_subhalos, kwargs_mc_subs = preset_concentration_models(concentration_model_subhalos,
                                                                 kwargs_concentration_model_subhalos)
    scatter_dex_subhalos = 0.2
    kwargs_mc_subs['scatter_dex'] = scatter_dex_subhalos
    concentration_model_subhalos = model_subhalos(**kwargs_mc_subs)

    model_fieldhalos, kwargs_mc_field = preset_concentration_models(concentration_model_fieldhalos,
                                                                    kwargs_concentration_model_fieldhalos)
    scatter_dex_fieldhalos = 0.2
    kwargs_mc_field['scatter_dex'] = scatter_dex_fieldhalos
    concentration_model_fieldhalos = model_fieldhalos(**kwargs_mc_field)
    if c_host is None:
        c_host = concentration_model_fieldhalos.nfw_concentration(10 ** log_m_host, z_lens)

    # SET THE TRUNCATION RADIUS FOR SUBHALOS AND FIELD HALOS
    kwargs_truncation_model_subhalos['lens_cosmo'] = pyhalo.lens_cosmo
    kwargs_truncation_model_fieldhalos['lens_cosmo'] = pyhalo.lens_cosmo

    model_subhalos, kwargs_trunc_subs = truncation_models(truncation_model_subhalos)
    kwargs_trunc_subs.update(kwargs_truncation_model_subhalos)
    if truncation_model_subhalos in ['TRUNCATION_GALACTICUS_KEELEY24', 'TRUNCATION_GALACTICUS']:
        kwargs_trunc_subs['c_host'] = c_host
    truncation_model_subhalos = model_subhalos(**kwargs_trunc_subs)

    model_fieldhalos, kwargs_trunc_field = truncation_models(truncation_model_fieldhalos)
    kwargs_trunc_field.update(kwargs_truncation_model_fieldhalos)
    truncation_model_fieldhalos = model_fieldhalos(**kwargs_trunc_field)

    # NOW THAT THE CLASSES ARE SPECIFIED, WE SORT THE KEYWORD ARGUMENTS AND CLASSES INTO LISTS
    population_model_list = ['SUBHALOS', 'LINE_OF_SIGHT']
    mass_function_class_list = [mass_function_model_subhalos, mass_function_model_fieldhalos]
    # check for log10 value of sigma_sub
    if log10_sigma_sub is not None:
        sigma_sub = 10 ** log10_sigma_sub
    kwargs_subhalos = {'log_mlow': log_mlow,
                       'log_mhigh': log_mhigh,
                       'm_pivot': 10 ** 8,
                       'power_law_index': shmf_log_slope,
                       'delta_power_law_index': delta_power_law_index,
                       'log_m_host': log_m_host,
                       'sigma_sub': sigma_sub,
                       'host_scaling_factor': host_scaling_factor,
                       'redshift_scaling_factor': redshift_scaling_factor,
                       'draw_poisson': draw_poisson}
    kwargs_los = {'log_mlow': log_mlow,
                       'log_mhigh': log_mhigh,
                  'LOS_normalization': LOS_normalization,
                       'm_pivot': 10 ** 8,
                       'delta_power_law_index': delta_power_law_index,
                       'log_m_host': log_m_host,
                    'draw_poisson': draw_poisson}
    kwargs_mass_function_list = [kwargs_subhalos, kwargs_los]

    # SET THE SPATIAL DISTRIBUTION MODELS FOR SUBHALOS AND FIELD HALOS:
    if subhalo_spatial_distribution == 'UNIFORM':
        subhalo_spatial_distribution = Uniform
        kwargs_subhalos_spatial = {'rmax2d_arcsec': cone_opening_angle_arcsec / 2,
                                   'geometry': geometry
                                   }
    elif subhalo_spatial_distribution == 'PROJECTED_NFW':
        subhalo_spatial_distribution = ProjectedNFW
        kwargs_subhalos_spatial = {'m_host': 10 ** log_m_host, 'zlens': z_lens, 'c_host': c_host,
                                   'rmax2d_arcsec': cone_opening_angle_arcsec / 2, 'r_core_units_rs': r_tidal,
                                   'lens_cosmo': pyhalo.lens_cosmo}
    else:
        raise Exception('subhalo spatial distribution must be either UNIFORM OR PROJECTED_NFW')
    fieldhalo_spatial_distribution = LensConeUniform
    spatial_distribution_class_list = [subhalo_spatial_distribution, fieldhalo_spatial_distribution]
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
    realization = pyhalo.render(population_model_list, mass_function_class_list, kwargs_mass_function_list,
                                     spatial_distribution_class_list, kwargs_spatial_distribution_list,
                                     geometry, mdef_subhalos, mdef_field_halos, kwargs_halo_model,
                                     two_halo_Lazar_correction, scale_2halo_boost_factor=1.0,
                                     nrealizations=1)[0]
    if mass_threshold_sis is not None:
        ext = RealizationExtensions(realization)
        realization = ext.SIS_injection(mass_threshold_sis, galaxy_model)
    if add_globular_clusters:
        ext = RealizationExtensions(realization)
        realization = ext.add_globular_clusters(**kwargs_globular_clusters)
    return realization

def CDMCorrelatedStructure(z_lens, z_source, log_mlow=6., log_mhigh=10.,
        concentration_model='LUDLOW2016', kwargs_concentration_model={},
        truncation_model='TRUNCATION_RN', kwargs_truncation_model={},
        cone_opening_angle_arcsec=6., log_m_host=13.3, LOS_normalization=1.0,
        delta_power_law_index=0.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None,
        scale_2halo_boost_factor=1.0, mass_threshold_sis=5*10**10):
    """

    :param z_lens:
    :param z_source:
    :param log_mlow:
    :param log_mhigh:
    :param concentration_model:
    :param kwargs_concentration_model:
    :param truncation_model:
    :param kwargs_truncation_model:
    :param cone_opening_angle_arcsec:
    :param log_m_host:
    :param LOS_normalization:
    :param delta_power_law_index:
    :param geometry_type:
    :param kwargs_cosmo:
    :param mass_threshold_sis: the mass threshold above which NFW profiles become SIS
    :return:
    """
    # FIRST WE CREATE AN INSTANCE OF PYHALO, WHICH SETS THE COSMOLOGY
    pyhalo = pyHalo(z_lens, z_source, kwargs_cosmo)
    # WE ALSO SPECIFY THE GEOMETRY OF THE RENDERING VOLUME
    geometry = Geometry(pyhalo.cosmology, z_lens, z_source,
                        cone_opening_angle_arcsec, geometry_type)

    # NOW WE SET THE MASS FUNCTION CLASSES FOR SUBHALOS AND FIELD HALOS
    # NOTE: MASS FUNCTION CLASSES SHOULD NOT BE INSTANTIATED HERE
    mass_function_model = ShethTormen

    # SET THE SPATIAL DISTRIBUTION MODELS FOR SUBHALOS AND FIELD HALOS:
    fieldhalo_spatial_distribution = LensConeUniform
    kwargs_spatial_distribution = {'cone_opening_angle': cone_opening_angle_arcsec, 'geometry': geometry}

    # set the density profile definition
    mdef_subhalos = 'TNFW'
    mdef_field_halos = 'TNFW'

    kwargs_concentration_model['cosmo'] = pyhalo.astropy_cosmo

    # SET THE CONCENTRATION-MASS RELATION MODEL
    model, kwargs_mc = preset_concentration_models(concentration_model, kwargs_concentration_model)
    concentration_model = model(**kwargs_mc)

    # SET THE TRUNCATION RADIUS MODEL
    kwargs_truncation_model['lens_cosmo'] = pyhalo.lens_cosmo
    model, kwargs_trunc = truncation_models(truncation_model)
    kwargs_trunc.update(kwargs_truncation_model)
    truncation_model = model(**kwargs_trunc)

    kwargs_halo_model = {'truncation_model_subhalos': truncation_model,
                         'concentration_model_subhalos': concentration_model,
                         'truncation_model_field_halos': truncation_model,
                         'concentration_model_field_halos': concentration_model,
                         'kwargs_density_profile': {}}
    kwargs_mass_function = {'log_mlow': log_mlow,
                  'log_mhigh': log_mhigh,
                  'LOS_normalization': LOS_normalization,
                  'm_pivot': 10 ** 8,
                  'delta_power_law_index': delta_power_law_index,
                  'log_m_host': log_m_host}
    population_model_list = ['TWO_HALO']
    mass_function_class_list = [mass_function_model]
    spatial_distribution_class_list = [fieldhalo_spatial_distribution]
    kwargs_spatial_distribution_list = [kwargs_spatial_distribution]
    kwargs_mass_function_list = [kwargs_mass_function]
    two_halo_Lazar_correction = True
    realization = pyhalo.render(population_model_list, mass_function_class_list, kwargs_mass_function_list,
                                     spatial_distribution_class_list, kwargs_spatial_distribution_list,
                                     geometry, mdef_subhalos, mdef_field_halos, kwargs_halo_model,
                                     two_halo_Lazar_correction, scale_2halo_boost_factor, nrealizations=1)[0]
    if mass_threshold_sis is not None:
        ext = RealizationExtensions(realization)
        realization = ext.SIS_injection(mass_threshold_sis)
    return realization

