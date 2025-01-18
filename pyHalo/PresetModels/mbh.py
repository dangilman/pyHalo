from pyHalo.PresetModels.cdm import CDM
from pyHalo.realization_extensions import RealizationExtensions

def CDM_plus_BH(z_lens,
                 z_source,
                 log10_mass_ratio,
                 log10_occupation_frac,
                 log10_mlow_halos_subres=5.0,
                 log10_min_mbh=4.5,
                 log10_mass_maximum=6.7,
                 sigma_sub=0.025,
                 log_mlow=6.,
                 log_mhigh=10.,
                 log10_sigma_sub=None,
                 shmf_log_slope=-1.9,
                 cone_opening_angle_arcsec=6.,
                 log_m_host=13.3,
                 LOS_normalization=1.0,
                 geometry_type='DOUBLE_CONE',
                 add_globular_clusters=False,
                 kwargs_globular_clusters=None):
    """
    Add a population of black holes to a CDM realization
    some literature:
    - https://arxiv.org/pdf/0709.0529 (evolution of Mbh seeds)
    :param z_lens: lens redshift
    :param z_source: source redshift
    :param log10_mass_ratio: the ratio of the black hole mass to the host halo mass
    :param log10_occupation_frac: the fraction of halos with a bh in them
    :param log10_mlow_halos_subres: the minimum halo mass that host black holes, can be smaller than the minimum
    halo mass rendered in the model (set by log_mlow)
    :param log10_min_mbh: the minimum black hole mass to render
    :param log10_mass_maximum: the maximum black hole mass to render
    :param sigma_sub: SHMF normalization
    :param log_mlow: log10 minimum DM halo mass to render
    :param log_mhigh: log10 maximum DM halo mass to render
    :param log10_sigma_sub: SHMF normalization
    :param shmf_log_slope: SHMF log-slope
    :param cone_opening_angle_arcsec: the opening angle of the lensing volume
    :param log_m_host: host halo mass
    :param LOS_normalization: amplitude of the LOS HMF relative to Sheth Tormen
    :param geometry_type: specifies the geometry of the rendering volume
    :param add_globular_clusters: bool; include a population of globular clusters
    :param kwargs_globular_clusters: keyword args for the GC population, see documentation in RealizationExtensions
    :return: a realization with CDM halos plus black hole seeds
    """

    cdm = CDM(z_lens, z_source, sigma_sub, log_mlow, log_mhigh,
              log10_sigma_sub, shmf_log_slope=shmf_log_slope, cone_opening_angle_arcsec=cone_opening_angle_arcsec,
              log_m_host=log_m_host, LOS_normalization=LOS_normalization, geometry_type=geometry_type)
    ext = RealizationExtensions(cdm)
    mbh = ext.add_black_holes(log10_mass_ratio,
                        10**log10_occupation_frac,
                        log10_mlow_halos_subres,
                        log10_min_mbh,
                        log_mlow,
                        log10_mass_maximum,
                        LOS_normalization)
    realization = cdm.join(mbh)
    if add_globular_clusters:
        ext = RealizationExtensions(realization)
        realization = ext.add_globular_clusters(**kwargs_globular_clusters)
    return realization
