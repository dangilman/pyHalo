import numpy as np
from pyHalo.Halos.HaloModels.powerlaw import PowerLawSubhalo, PowerLawFieldHalo
from pyHalo.single_realization import Realization


class RealizationExtensions(object):

    """
    This class supports operations that modify individual instances of the class Realization
    (see pyHalo.single_realization).
    """

    def __init__(self, realization):

        """

        :param realization: an instance of Realization
        """

        self._realization = realization

    def change_mass_definition(self, mdef, new_mdef, kwargs_new):

        kwargs_realization = self._realization._prof_params
        kwargs_realization.update(kwargs_new)
        halos = self._realization.halos
        new_halos = []

        if new_mdef == 'coreTNFW':
            from pyHalo.Halos.HaloModels.coreTNFW import coreTNFWSubhalo, coreTNFWFieldHalo
        else:
            raise Exception('changing to mass definition '+new_mdef + ' not implemented')

        for halo in halos:
            if halo.mdef == mdef:
                if halo.is_subhalo:
                    new_halo = coreTNFWSubhalo.fromTNFW(halo, kwargs_realization)
                else:
                    new_halo = coreTNFWFieldHalo.fromTNFW(halo, kwargs_realization)
                new_halos.append(new_halo)
            else:
                new_halos.append(halo)

        lens_cosmo = self._realization.lens_cosmo
        prof_params = self._realization._prof_params
        msheet_correction = self._realization._mass_sheet_correction
        rendering_classes = self._realization.rendering_classes
        rendering_center_x, rendering_center_y = self._realization.rendering_center

        return Realization.from_halos(new_halos, lens_cosmo, prof_params,
                                      msheet_correction, rendering_classes,
                                      rendering_center_x, rendering_center_y,
                                      self._realization.geometry)



    def find_core_collapsed_halos(self, time_scale_function, velocity_dispersion_function,
                                  cross_section, t_sub=10., t_field=100., t_sub_range=2, t_field_range=2.,
                                  model_type='TCHANNEL'):

        """
        :param time_scale_function: a function that computes the characteristic timescale for SIDM halos. This function
        must take as an input the NFW halo density normalization, velocity dispersion, and cross section class,
        and it must return a timescale (t_scale)
        :param velocity_dispersion_function: a function that computes the central velocity disperion of the halo

        It must be callable as:
        v = velocity_dispersion_function(halo_mass, redshift, delta_c_over_c, model_type, additional_keyword_arguments)

        where model_type is a string (see for example the function solve_sigmav_with_interpolation in sidmpy.py)

        :param velocity_averaged_cross_section_function: a function that returns the velocity averaged interaction
        cross section [cm^2 / gram * km/sec]
        :param t_sub: sets the timescale for subhalo core collapse; subhalos collapse at t_sub * t_scale
        :param t_field: sets the timescale for field halo core collapse; field halos collapse at t_field * t_scale
        :param t_sub_range: halos begin to core collapse (probability = 0) at t_sub - t_sub_range, and all are core
        collapsed by t = t_sub + t_sub_range (probability = 1)
        :param t_field_range: field halos begin to core collapse (probability = 0) at t_field - t_field_range, and all
        are core collapsed by t = t_field + t_field_range (probability = 1)
        :param model_type: specifies the cross section model to use when computing the solution to the velocity
        dispersion of the halo
        :return: indexes of halos that are core collapsed given
        """
        inds = []
        for i, halo in enumerate(self._realization.halos):

            if halo.mdef not in ['NFW', 'TNFW', 'coreTNFW']:
                continue

            # fit calibrated from the NFW velocity dispersion inside rs between 10^6 and 10^10
            # coeffs = [0.31575757, -1.74259129]
            # log_vrms = coeffs[0] * np.log10(halo.mass) + coeffs[1]
            # v_rms = 10 ** log_vrms

            concentration = halo.profile_args[0]
            rhos, rs = halo.params_physical['rhos'], halo.params_physical['rs']
            median_concentration = self._realization.lens_cosmo.NFW_concentration(halo.mass, halo.z, scatter=False)
            delta_c_over_c = 1 - concentration/median_concentration
            v_rms = velocity_dispersion_function(halo.mass, halo.z, delta_c_over_c, model_type, cross_section.kwargs)
            timescale = time_scale_function(rhos, v_rms, cross_section)

            if halo.is_subhalo:
                tcollapse_min = timescale * t_sub / t_sub_range
                tcollapse_max = timescale * t_sub * t_sub_range
            else:
                tcollapse_min = timescale * t_field / t_field_range
                tcollapse_max = timescale * t_field * t_field_range

            halo_age = self._realization.lens_cosmo.cosmo.halo_age(halo.z)

            if halo_age > tcollapse_max:
                p = 1.
            elif halo_age < tcollapse_min:
                p = 0.
            else:
                p = (halo_age - tcollapse_min) / (tcollapse_max - tcollapse_min)

            u = np.random.rand()
            if p >= u:
                inds.append(i)

        return inds

    def add_core_collapsed_halos(self, indexes, **kwargs_halo):

        """
        This function turns NFW halos in a realization into profiles modeled as PseudoJaffe profiles
        with 1/r^2 central density profiles with the same total mass as the original NFW
        profile.

        :param indexes: the indexes of halos in the realization to transform into PsuedoJaffe profiles
        :param log_slope_halo: the logarithmic slope of the collapsed halos
        :return: A new instance of Realization where the halos indexed by indexes
        in the original realization have their mass definitions changed to PsuedoJaffe
        """

        halos = self._realization.halos
        new_halos = []

        collapsed_subhalo_profile = PowerLawSubhalo
        collapsed_field_profile = PowerLawFieldHalo

        for i, halo in enumerate(halos):

            if i in indexes:

                halo._args.update(kwargs_halo)
                if halo.is_subhalo:
                    new_halo = collapsed_subhalo_profile(halo.mass, halo.x, halo.y, halo.r3d, 'SPL_CORE',
                                                 halo.z, True, halo.lens_cosmo, halo._args, halo.unique_tag)
                else:
                    new_halo = collapsed_field_profile(halo.mass, halo.x, halo.y, halo.r3d, 'SPL_CORE',
                                                 halo.z, False, halo.lens_cosmo, halo._args, halo.unique_tag)
                new_halos.append(new_halo)

            else:
                new_halos.append(halo)

        lens_cosmo = self._realization.lens_cosmo
        prof_params = self._realization._prof_params
        msheet_correction = self._realization._mass_sheet_correction
        rendering_classes = self._realization.rendering_classes
        rendering_center_x, rendering_center_y = self._realization.rendering_center

        return Realization.from_halos(new_halos, lens_cosmo, prof_params,
                                      msheet_correction, rendering_classes,
                                      rendering_center_x, rendering_center_y,
                                      self._realization.geometry)
