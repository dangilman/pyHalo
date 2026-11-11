import numpy as np
from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo
from pyHalo.Halos.tidal_truncation import TruncationRN
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.utilities import generate_lens_plane_redshifts


class ConvergenceSheetRescaling(object):
    """
    A callable function of redshift that rescales the amplitude of the negative convergence sheets along the line of
    sight. Pass an instance as kappa_scale in kwargs_mass_sheet, e.g.
    realization.lensing_quantities(kwargs_mass_sheet={'kappa_scale': rescaling, ...})

    For halo centers distributed uniformly inside a circle of radius R, the convergence at the center is

    N <M_2D(<R)> / (pi R^2)

    M_2D(<R) can be written as pi*R*alpha(R), where alpha(R) is the deflection angle at R. The rescaling factor is

    <M_2D(<R)> / <M_200> = int dm dN/dm alpha_m(R) / R / (int dm dN/dm m / (Sigma_crit pi R^2))

    summed over all line-of-sight populations at each plane. It is less than one where halos are large compared to the
    rendering area, and greater than one where the rendering area contains most of the truncated halo profile
    """

    def __init__(self, redshifts, rescaling_factor):
        """

        :param redshifts: redshifts of the lens planes
        :param rescaling_factor: the factor that multiplies the default convergence sheet at each redshift
        """
        self.redshifts = np.array(redshifts)
        self.rescaling_factor = np.array(rescaling_factor)

    def __call__(self, z):
        """
        Evaluates the rescaling factor at redshift z (interpolated between lens planes)
        :param z: redshift
        :return: the factor that multiplies the default convergence sheet at z
        """
        return np.interp(z, self.redshifts, self.rescaling_factor)

    @classmethod
    def from_realization(cls, realization, log_mlow=None, log_mhigh=None, n_mass=25):
        """
        Computes the rescaling with the line-of-sight mass functions, rendering geometry, concentration-mass relation,
        and truncation model of an existing realization. For realizations with more than one line-of-sight population,
        compute the sheets with use_class_mass_ranges=True so that each population subtracts its own mass range.

        :param realization: an instance of Realization
        :param log_mlow: log10 of the minimum halo mass subtracted by the sheets (log_mlow_sheets); if None, uses the
        mass range of each line-of-sight population (use_class_mass_ranges=True)
        :param log_mhigh: log10 of the maximum halo mass subtracted by the sheets (log_mhigh_sheets); if None, uses the
        mass range of each line-of-sight population
        :param n_mass: number of mass bins used to integrate over each halo mass function at each plane
        :return: an instance of ConvergenceSheetRescaling
        """
        los_classes = [rc for rc in realization.rendering_classes if rc is not None and rc.name == 'LINE_OF_SIGHT']
        if len(los_classes) == 0:
            raise Exception('realization has no LINE_OF_SIGHT rendering classes')
        # the rescaling is computed with TNFW profiles; check the field halos actually rendered in the realization

        mass_function_components = []
        for rendering_class in los_classes:
            # same mass range and mass function model as LineOfSight.convergence_sheet_correction
            def component(z, delta_z, rendering_class=rendering_class):
                log_mlow_z, log_mhigh_z = rendering_class._redshift_dependent_mass_range(
                    z, rendering_class._kwargs_mass_function['log_mlow'] if log_mlow is None else log_mlow,
                    rendering_class._kwargs_mass_function['log_mhigh'] if log_mhigh is None else log_mhigh)
                mass_function = rendering_class._get_mass_function_model(z, delta_z, log_mlow=log_mlow_z,
                                                                         log_mhigh=log_mhigh_z)
                return log_mlow_z, log_mhigh_z, mass_function
            mass_function_components.append(component)
        redshifts, rescaling_factor = rescaling_factor_line_of_sight(
            realization.geometry, realization.lens_cosmo, los_classes[0]._lens_plane_redshifts,
            los_classes[0]._delta_z_list, mass_function_components,
            realization.kwargs_halo_model['concentration_model_field_halos'],
            realization.kwargs_halo_model['truncation_model_field_halos'], n_mass)
        return ConvergenceSheetRescaling(redshifts, rescaling_factor)


class ConvergenceSheetRescalingShethTormen(ConvergenceSheetRescaling):
    """
    The rescaling of the line-of-sight convergence sheets (see ConvergenceSheetRescaling) for halos drawn from the
    Sheth-Tormen mass function, with TNFW profiles truncated at r_N. The default halo model is that of the CDM preset
    model (truncation at r_50 and the Ludlow et al. (2016) concentration-mass relation).
    """

    def __init__(self, z_lens, z_source, log_mlow=6.0, log_mhigh=10.0, LOS_normalization=1.0,
                 cone_opening_angle_arcsec=6.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None,
                 lens_plane_spacing=None, concentration_model='LUDLOW2016', kwargs_concentration_model=None,
                 LOS_truncation_factor=50, n_mass=25):
        """

        :param z_lens: main deflector redshift
        :param z_source: source redshift
        :param log_mlow: log10 of the minimum halo mass subtracted by the sheets; a number, or a list with one entry
        for each line-of-sight population (e.g. [6.0, 8.0] for CDMBinned)
        :param log_mhigh: log10 of the maximum halo mass subtracted by the sheets; a number or a list
        :param LOS_normalization: amplitude of the line-of-sight halo mass function; a number or a list. With a single
        population the amplitude cancels in the rescaling, with several it sets their relative weight
        :param cone_opening_angle_arcsec: the opening angle of the rendering volume in arcsec; should match the value
        used to create the realization
        :param geometry_type: the geometry of the rendering volume
        :param kwargs_cosmo: keyword arguments for the Cosmology class
        :param lens_plane_spacing: sets the spacing between LOS lens planes, if None reverts to default value of 0.02
        :param concentration_model: the concentration-mass relation applied to field halos, either the name of a
        model in concentration_models.py or an instance of a concentration class (the median relation is used)
        :param kwargs_concentration_model: keyword arguments for the concentration model if specified by name
        :param LOS_truncation_factor: field halos are truncated at r_N with N = LOS_truncation_factor
        :param n_mass: number of mass bins used to integrate over each halo mass function at each plane
        """
        pyhalo = pyHalo(z_lens, z_source, kwargs_cosmo)
        lens_cosmo = pyhalo.lens_cosmo
        geometry = Geometry(pyhalo.cosmology, z_lens, z_source, cone_opening_angle_arcsec, geometry_type)
        if isinstance(concentration_model, str):
            kwargs_model = {} if kwargs_concentration_model is None else dict(kwargs_concentration_model)
            kwargs_model['cosmo'] = pyhalo.astropy_cosmo
            model, kwargs_model = preset_concentration_models(concentration_model, kwargs_model)
            concentration_class = model(**kwargs_model)
        else:
            concentration_class = concentration_model
        truncation_class = TruncationRN(lens_cosmo, LOS_truncation_factor)
        lens_plane_redshifts, delta_z_list = generate_lens_plane_redshifts(z_lens, z_source, lens_plane_spacing)

        log_mlow_list, log_mhigh_list, normalization_list = np.broadcast_arrays(np.atleast_1d(log_mlow),
                                                                               np.atleast_1d(log_mhigh),
                                                                               np.atleast_1d(LOS_normalization))
        mass_function_components = []
        for log_mlow_i, log_mhigh_i, normalization_i in zip(log_mlow_list, log_mhigh_list, normalization_list):
            kwargs_mass_function = {'log_mlow': float(log_mlow_i), 'log_mhigh': float(log_mhigh_i),
                                    'LOS_normalization': float(normalization_i), 'm_pivot': 10 ** 8,
                                    'delta_power_law_index': 0.0, 'draw_poisson': False}
            def component(z, delta_z, kwargs_mass_function=kwargs_mass_function):
                return (kwargs_mass_function['log_mlow'], kwargs_mass_function['log_mhigh'],
                        ShethTormen.from_redshift(z, delta_z, geometry, kwargs_mass_function))
            mass_function_components.append(component)
        redshifts, rescaling_factor = rescaling_factor_line_of_sight(geometry, lens_cosmo, lens_plane_redshifts,
                                                                     delta_z_list, mass_function_components,
                                                                     concentration_class, truncation_class, n_mass)
        super(ConvergenceSheetRescalingShethTormen, self).__init__(redshifts, rescaling_factor)


def rescaling_factor_line_of_sight(geometry, lens_cosmo, lens_plane_redshifts, delta_z_list,
                                   mass_function_components, concentration_class, truncation_class, n_mass=25):
    """
    Computes the ratio between the expected convergence of line-of-sight halos at the center of the rendering area and
    the default convergence sheets (mean mass in M_200 per unit area) at each lens plane

    :param geometry: an instance of Geometry
    :param lens_cosmo: an instance of LensCosmo
    :param lens_plane_redshifts: redshifts of the lens planes
    :param delta_z_list: thickness of each lens plane
    :param mass_function_components: a list of functions (z, delta_z) -> (log_mlow, log_mhigh, mass function model),
    one for each line-of-sight population
    :param concentration_class: the concentration-mass relation (the median relation is used)
    :param truncation_class: the truncation model for field halos
    :param n_mass: number of mass bins used to integrate over each halo mass function at each plane
    :return: redshifts, rescaling factor at each redshift
    """
    median_concentration = _MedianConcentration(concentration_class)
    nodes, weights = np.polynomial.legendre.leggauss(n_mass)
    redshifts, rescaling = [], []
    for z, delta_z in zip(lens_plane_redshifts, delta_z_list):
        radius = 0.5 * geometry.cone_opening_angle * geometry.rendering_scale(z)
        area = geometry.angle_to_physical_area(0.5 * geometry.cone_opening_angle, z)
        kappa, kappa_default = 0.0, 0.0
        for component in mass_function_components:
            log_mlow, log_mhigh, mass_function = component(z, delta_z)
            if log_mlow >= log_mhigh or mass_function.first_moment == 0:
                continue
            # Gauss-Legendre nodes in log10(m) never fall on the edges of the mass range, where binned
            # concentration-mass relations are undefined
            log_m = 0.5 * (log_mhigh - log_mlow) * nodes + 0.5 * (log_mhigh + log_mlow)
            m = 10 ** log_m
            alpha = [TNFWFieldHalo(mi, 0.0, 0.0, None, z, False, lens_cosmo, {}, truncation_class,
                                   median_concentration, 0).deflection_angle(radius) for mi in m]
            dn_dlogm = mass_function.evaluate(m) * m * np.log(10)
            kappa += 0.5 * (log_mhigh - log_mlow) * np.sum(weights * dn_dlogm * np.array(alpha)) / radius
            kappa_default += mass_function.first_moment / lens_cosmo.sigma_crit_mass(z, area)
        if kappa_default > 0:
            redshifts.append(z)
            rescaling.append(kappa / kappa_default)
    return np.array(redshifts), np.array(rescaling)


class _MedianConcentration(object):
    """
    Evaluates the median concentration-mass relation of a concentration class, so that one halo represents each mass
    """

    def __init__(self, concentration_class):
        self._concentration_class = concentration_class

    def nfw_concentration(self, m, z):
        if hasattr(self._concentration_class, '_cdm_concentration'):
            # models with a turnover rescale a CDM relation by a suppression factor
            c_cdm = self._concentration_class._cdm_concentration.nfw_concentration(m, z, force_no_scatter=True)
            return max(c_cdm * self._concentration_class.suppression(m, z),
                       self._concentration_class._universal_minimum)
        return self._concentration_class.nfw_concentration(m, z, force_no_scatter=True)
