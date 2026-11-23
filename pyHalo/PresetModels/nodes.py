from copy import deepcopy
from pyHalo.PresetModels.cdm import CDM
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.Rendering.MassFunctions.node_ratio import NodeRatioMassFunction, NodeRatioSubhaloMassFunction

__all__ = ['HMFConcentrationNodes']

# keyword arguments of the CDM preset that this preset sets
_RESERVED_KEYS = ['mass_function_model_fieldhalos', 'kwargs_mass_function_fieldhalos',
                  'mass_function_model_subhalos', 'kwargs_mass_function_subhalos',
                  'concentration_model_subhalos', 'kwargs_concentration_model_subhalos',
                  'concentration_model_fieldhalos', 'kwargs_concentration_model_fieldhalos']


def HMFConcentrationNodes(z_lens, z_source, log10_n_ratio_6=0.0, log10_n_ratio_8=0.0, log10_c_ratio_6=0.0,
                          log10_c_ratio_8=0.0, log10_c_ratio_10=0.0, log10_n_ratio_10=0.0, log10_n_ratio_floor=-2.0,
                          concentration_model_cdm='LUDLOW2016', kwargs_concentration_model_cdm=None, **kwargs):
    """
    Realizations of dark matter structure in which the halo mass function and the concentration-mass relation deviate
    from CDM by ratios specified at three halo masses (nodes): 10^6, 10^8 and 10^10 M_sun. All other ingredients are
    those of the CDM preset, which this preset reproduces when every node is zero.

    The mass function ratio R(m) = n / n_CDM has log10 R = log10_n_ratio_6 and log10_n_ratio_8 at 10^6 and 10^8 M_sun,
    and log10 R = log10_n_ratio_10 at 10^10 M_sun (zero by default; the calibrated prior keeps this node fixed at CDM,
    because power spectrum features at k > 10 / Mpc barely change the mass function at 10^10 M_sun). The concentration
    ratio c / c_CDM has log10 values log10_c_ratio_6, log10_c_ratio_8
    and log10_c_ratio_10 at the three nodes. Between the nodes each log10 ratio is a natural cubic spline in log10(M);
    outside the nodes it is held at the end values, and log10 R is floored at log10_n_ratio_floor (see
    pyHalo.node_ratios). The ratios do not depend on redshift, and apply to:

    1) field halos and the two-halo term: the Sheth-Tormen mass function of the CDM preset times R(m) at each lens plane
    2) subhalos: the CDM power law of the subhalo mass function at infall times R(m); sigma_sub (or log10_sigma_sub,
    log10_dNdA) keeps its CDM meaning, so the amplitude of the subhalo mass function at 10^8 M_sun is sigma_sub * R(10^8)
    3) concentrations of subhalos and field halos: c = c_CDM * 10^(spline), with the scatter drawn for c_CDM

    The negative convergence sheets subtract the mass in halos of the modified mass functions.

    The ratios can be sampled from a prior calibrated on linear power spectra with features beyond k = 10 / Mpc (a
    running of the spectral index, or bumps), using the Benson et al. (2026) mass function and the Ludlow et al. (2016)
    concentration-mass relation to compute the response of each node.

    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param log10_n_ratio_6: log10(n / n_CDM) at 10^6 M_sun
    :param log10_n_ratio_8: log10(n / n_CDM) at 10^8 M_sun
    :param log10_c_ratio_6: log10(c / c_CDM) at 10^6 M_sun
    :param log10_c_ratio_8: log10(c / c_CDM) at 10^8 M_sun
    :param log10_c_ratio_10: log10(c / c_CDM) at 10^10 M_sun
    :param log10_n_ratio_10: log10(n / n_CDM) at 10^10 M_sun (default zero)
    :param log10_n_ratio_floor: the smallest value of log10(n / n_CDM) at any mass; None for no floor
    :param concentration_model_cdm: the CDM concentration-mass relation that the ratio multiplies, as a name in
    concentration_models.py (default LUDLOW2016, as in the CDM preset)
    :param kwargs_concentration_model_cdm: keyword arguments for the CDM concentration-mass relation
    :param kwargs: keyword arguments of the CDM preset (sigma_sub, log_mlow, log_mhigh, ...)
    :return: a realization of dark matter halos
    """
    for key in _RESERVED_KEYS:
        if key in kwargs:
            raise TypeError(key + ' is set by the HMFConcentrationNodes preset')
    log10_n_ratio_nodes = [log10_n_ratio_6, log10_n_ratio_8, log10_n_ratio_10]
    log10_c_ratio_nodes = [log10_c_ratio_6, log10_c_ratio_8, log10_c_ratio_10]
    concentration_cdm_class, kwargs_cdm = preset_concentration_models(concentration_model_cdm,
                                                                      kwargs_concentration_model_cdm)
    kwargs_mass_function = {'log10_n_ratio_nodes': log10_n_ratio_nodes, 'log10_n_ratio_floor': log10_n_ratio_floor}
    kwargs_concentration = {'concentration_cdm_class': concentration_cdm_class, 'kwargs_cdm': kwargs_cdm,
                            'log10_c_ratio_nodes': log10_c_ratio_nodes}
    return CDM(z_lens, z_source,
               mass_function_model_fieldhalos=NodeRatioMassFunction,
               kwargs_mass_function_fieldhalos=deepcopy(kwargs_mass_function),
               mass_function_model_subhalos=NodeRatioSubhaloMassFunction,
               kwargs_mass_function_subhalos=deepcopy(kwargs_mass_function),
               concentration_model_subhalos='NODE_RATIO',
               kwargs_concentration_model_subhalos=deepcopy(kwargs_concentration),
               concentration_model_fieldhalos='NODE_RATIO',
               kwargs_concentration_model_fieldhalos=deepcopy(kwargs_concentration), **kwargs)
