import re
import numpy as np
from pyHalo.pyhalo import pyHalo
from pyHalo.PresetModels.cdm import CDM
from pyHalo.Cosmology.power_spectrum import LinearPowerSpectrum
from pyHalo.Cosmology.binned_power_spectrum import binned_linear_power_spectrum, lcdm_power_spectrum
from pyHalo.Rendering.MassFunctions.power_spectrum import PowerSpectrumMassFunction, \
    PowerSpectrumSubhaloMassFunction, MULTIPLICITY_FUNCTION_PARAMS

__all__ = ['BinnedPowerSpectrum']

_AMPLITUDE_KEY = re.compile(r'^pk_amplitude_(\d+)$')
# keyword arguments of the CDM preset that this preset sets
_RESERVED_KEYS = ['mass_function_model_fieldhalos', 'kwargs_mass_function_fieldhalos',
                  'mass_function_model_subhalos', 'kwargs_mass_function_subhalos',
                  'concentration_model_subhalos', 'kwargs_concentration_model_subhalos',
                  'concentration_model_fieldhalos', 'kwargs_concentration_model_fieldhalos']


def BinnedPowerSpectrum(z_lens, z_source, pk_amplitudes=None, pk_k_min=5.0, pk_k_max=200.0, sigma_sub=0.025,
                        log10_sigma_sub=None, log10_dNdA=None, shmf_z_eval_offset=3.3,
                        multiplicity_function='BENSON2026', kwargs_cosmo=None, **kwargs):
    """
    Realizations of dark matter structure for a linear matter power spectrum that deviates from LCDM in logarithmically
    spaced bins of wavenumber (see pyHalo.Cosmology.binned_power_spectrum). The field halo, two-halo and subhalo mass
    functions and the concentration-mass relation of subhalos and field halos are computed from this power spectrum
    for each realization; all other ingredients are those of the CDM preset. The concentration-mass relation is always
    that of Ludlow et al. (2016) computed from the binned power spectrum (LUDLOW2016_PK), for subhalos and field halos.

    The bin amplitudes a_i set log10(P / P_LCDM) = a_i - 1 at the center of bin i, so a_i = 1 is LCDM and a_i = 1.3 is
    10^0.3 = 1.995 times LCDM. They can be passed either as a sequence (pk_amplitudes) or one at a time as keyword
    arguments pk_amplitude_1, pk_amplitude_2, ... (for example, when each amplitude is sampled as its own parameter).

    The field halo mass function is that of Benson et al. (2026) computed from the power spectrum (BENSON_PK). The
    subhalo mass function at infall is the power law of the CDM preset times the ratio R(m) of this mass function to
    its LCDM counterpart, evaluated at z_eval = z_lens + shmf_z_eval_offset (as in WDMGeneral), with the amplitude at
    10^8 M_sun set by

    sigma_sub_effective = sigma_sub * A_los / A_los_cdm

    where A_los / A_los_cdm = R(10^8 M_sun, z_eval) is the amplitude of the field halo mass function at 10^8 M_sun
    relative to LCDM; sigma_sub_effective = sigma_sub in LCDM. The same factor rescales log10_sigma_sub or log10_dNdA
    when those are used instead of sigma_sub.

    Notes:
    1) In LCDM, the Benson et al. (2026) field halo mass function lies 10-20% below the Sheth-Tormen mass function of
    the CDM preset; set multiplicity_function='SHETH_TORMEN' for Sheth-Tormen computed from the power spectrum
    2) The boost to the abundance of halos around the main deflector from the two-halo term is computed with the LCDM
    correlation function; for bins at k > 5 / Mpc it changes by less than 1% for amplitudes up to 1.6 (4 times LCDM),
    and by about 3% when every bin has an amplitude of 2.3 (20 times LCDM)

    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param pk_amplitudes: the amplitude of each bin; the number of bins is len(pk_amplitudes)
    :param pk_k_min: the smallest wavenumber of the first bin [1/Mpc]
    :param pk_k_max: the largest wavenumber of the last bin [1/Mpc]
    :param sigma_sub: the amplitude of the subhalo mass function at 10^8 M_sun in LCDM [# of halos / kpc^2]
    :param log10_sigma_sub: log10 of sigma_sub; replaces sigma_sub if specified
    :param log10_dNdA: log10 of the amplitude of the subhalo mass function at 10^8 M_sun in LCDM, without the scaling
    with host halo mass and redshift [# of halos / kpc^2]; replaces sigma_sub if specified
    :param shmf_z_eval_offset: the subhalo mass function is computed at z_lens + shmf_z_eval_offset; the default is the
    median infall redshift offset of the InfallDistributionHybrid model, as in WDMGeneral
    :param multiplicity_function: the multiplicity function of the halo mass functions; BENSON2026 or SHETH_TORMEN
    :param kwargs_cosmo: keyword arguments that specify the cosmology (see pyHalo.Cosmology.cosmology)
    :param kwargs: the bin amplitudes pk_amplitude_1, pk_amplitude_2, ... (if pk_amplitudes is None), and keyword
    arguments of the CDM preset
    :return: a realization of dark matter halos
    """
    amplitudes, kwargs_cdm = _bin_amplitudes(pk_amplitudes, kwargs)
    if multiplicity_function not in MULTIPLICITY_FUNCTION_PARAMS:
        raise ValueError('multiplicity_function must be one of ' + str(list(MULTIPLICITY_FUNCTION_PARAMS.keys())))
    for key in _RESERVED_KEYS:
        if key in kwargs_cdm:
            raise TypeError(key + ' is set by the BinnedPowerSpectrum preset')
    # the same cosmology the CDM preset creates for the realization
    cosmology = pyHalo(z_lens, z_source, kwargs_cosmo).cosmology
    power_spectrum = binned_linear_power_spectrum(cosmology, amplitudes, pk_k_min, pk_k_max)
    power_spectrum_lcdm = LinearPowerSpectrum(*lcdm_power_spectrum(cosmology), cosmology)

    # the subhalo mass function at infall, and the rescaling of its amplitude at 10^8 M_sun
    z_eval = z_lens + shmf_z_eval_offset
    sigma_sub_rescaling = sigma_sub_rescaling_factor(power_spectrum, power_spectrum_lcdm, z_eval,
                                                     multiplicity_function)
    sigma_sub = sigma_sub * sigma_sub_rescaling
    if log10_sigma_sub is not None:
        log10_sigma_sub = log10_sigma_sub + np.log10(sigma_sub_rescaling)
    if log10_dNdA is not None:
        log10_dNdA = log10_dNdA + np.log10(sigma_sub_rescaling)
    kwargs_mass_function_subhalos = {'power_spectrum': power_spectrum,
                                     'power_spectrum_reference': power_spectrum_lcdm,
                                     'z_eval': z_eval, 'multiplicity_function': multiplicity_function}
    kwargs_mass_function_fieldhalos = {'power_spectrum': power_spectrum, 'multiplicity_function': multiplicity_function}

    return CDM(z_lens, z_source, sigma_sub=sigma_sub, log10_sigma_sub=log10_sigma_sub, log10_dNdA=log10_dNdA,
               concentration_model_subhalos='LUDLOW2016_PK',
               kwargs_concentration_model_subhalos={'power_spectrum': power_spectrum},
               concentration_model_fieldhalos='LUDLOW2016_PK',
               kwargs_concentration_model_fieldhalos={'power_spectrum': power_spectrum},
               kwargs_cosmo=kwargs_cosmo, mass_function_model_fieldhalos=PowerSpectrumMassFunction,
               kwargs_mass_function_fieldhalos=kwargs_mass_function_fieldhalos,
               mass_function_model_subhalos=PowerSpectrumSubhaloMassFunction,
               kwargs_mass_function_subhalos=kwargs_mass_function_subhalos, **kwargs_cdm)


def sigma_sub_rescaling_factor(power_spectrum, power_spectrum_lcdm, z_eval, multiplicity_function='BENSON2026',
                               m_pivot=10 ** 8):
    """
    The factor A_los / A_los_cdm that rescales sigma_sub: the amplitude of the field halo mass function at m_pivot and
    z_eval computed from the power spectrum, relative to the one computed from the LCDM power spectrum

    :param power_spectrum: an instance of LinearPowerSpectrum
    :param power_spectrum_lcdm: the LinearPowerSpectrum of LCDM
    :param z_eval: the redshift at which the mass functions are evaluated
    :param multiplicity_function: BENSON2026 (default) or SHETH_TORMEN
    :param m_pivot: the halo mass at which the amplitudes are evaluated [M_sun]
    :return: A_los / A_los_cdm
    """
    ratio = PowerSpectrumSubhaloMassFunction.mass_function_ratio(np.array([m_pivot]), z_eval, power_spectrum,
                                                                 power_spectrum_lcdm, multiplicity_function)
    return float(ratio[0])


def _bin_amplitudes(pk_amplitudes, kwargs):
    """
    Reads the bin amplitudes from pk_amplitudes, or from keyword arguments pk_amplitude_1, pk_amplitude_2, ...
    :return: the amplitudes, and the remaining keyword arguments
    """
    kwargs_cdm = {}
    per_bin = {}
    for key, value in kwargs.items():
        match = _AMPLITUDE_KEY.match(key)
        if match is None:
            kwargs_cdm[key] = value
        else:
            per_bin[int(match.group(1))] = value
    if pk_amplitudes is not None:
        if len(per_bin) > 0:
            raise ValueError('specify the bin amplitudes either with pk_amplitudes or with pk_amplitude_1, '
                             'pk_amplitude_2, ..., not both')
        return list(pk_amplitudes), kwargs_cdm
    if len(per_bin) == 0:
        raise ValueError('the bin amplitudes must be specified with pk_amplitudes or with pk_amplitude_1, '
                         'pk_amplitude_2, ...')
    if sorted(per_bin.keys()) != list(range(1, len(per_bin) + 1)):
        raise ValueError('the bin amplitudes pk_amplitude_i must be numbered consecutively from i = 1; got ' +
                         str(sorted(per_bin.keys())))
    return [per_bin[i] for i in range(1, len(per_bin) + 1)], kwargs_cdm
