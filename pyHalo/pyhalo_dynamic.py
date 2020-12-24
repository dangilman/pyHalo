from pyHalo.Rendering.Field.PowerLaw.powerlaw_dynamic import LOSPowerLawDynamic
from pyHalo.Rendering.Field.Delta.delta_dynamic import LOSDeltaDynamic
from pyHalo.Rendering.Main.mainlens_dynamic import MainLensPowerLawDynamic
from pyHalo.pyhalo_base import pyHaloBase
from pyHalo.single_realization import Realization
from pyHalo.Rendering.render import render_los_dynamic, render_main_dynamic
from pyHalo.Cosmology.geometry import Geometry
from scipy.interpolate import interp1d
from pyHalo.defaults import set_default_kwargs
from copy import deepcopy

class pyHaloDynamic(pyHaloBase):

    def __init__(self, zlens, zsource, cosmology_kwargs={},
                 kwargs_halo_mass_function={}):

        """
        This class fits lens models to data while adding dark matter halos along the line of sight.

        It should only be used in conjuction with the DynamicOptimization class in LenstronomyWrapper.

        In fact, the only routine the user should call directly is 'init'

        :param zlens: lens redshift
        :param zsource: source redshift
        :param cosmology_kwargs:
        keyword arguments for 'Cosmology' class. See documentation in cosmology.py
        :param halo_mass_function_args:
        keyword arguments for 'LensingMassFunction' class. See documentation in lensing_mass_function.py
        :param kwargs_massfunc:
        keyword arguments
        """

        self._rendering_class_main = [None, None]
        super(pyHaloDynamic, self).__init__(zlens, zsource, cosmology_kwargs, kwargs_halo_mass_function)

    def reset(self, zlens, zsource):

        self._rendering_class_main = [None, None]
        self.reset_redshifts(zlens, zsource)

    def render_dynamic(self, realization_type, kwargs_realization, rendering_geometry, realization_start,
                       guiding_center_x_interp, guiding_center_y_interp, aperture_radius, verbose=False,
                       include_mass_sheet_correction=False, global_render=True):

        """

        :param realization_type:
        :param kwargs_realization:
        :param rendering_geometry:
        :param realization_start:
        :param guiding_center_x_interp:
        :param guiding_center_y_interp:
        :param aperture_radius:
        :param verbose:
        :param include_mass_sheet_correction:
        :param global_render:
        :return:
        """

        lens_plane_redshifts, delta_zs = self.lens_plane_redshifts(kwargs_realization[0])

        kwargs_init = set_default_kwargs(kwargs_realization, True, self.zsource)

        kwargs_realization = deepcopy(kwargs_init)

        for j, (x_position_interp, y_position_interp) in enumerate(zip(guiding_center_x_interp, guiding_center_y_interp)):

            if realization_type == 'main_lens' or realization_type == 'composite_powerlaw':
                realization_subs, rendering_class_subs = self._render_dynamic_main_single(kwargs_realization,
                                                                                          x_position_interp,
                                                                                          y_position_interp,
                                                                                          aperture_radius,
                                                                                          guiding_center_x_interp,
                                                                                          guiding_center_y_interp,
                                                                                          include_mass_sheet_correction,
                                                                                          rendering_geometry)

            if realization_type == 'line_of_sight' or realization_type == 'composite_powerlaw':
                realization_LOS, rendering_class_LOS = self._render_dynamic_los(kwargs_realization,
                                                                                x_position_interp,
                                                                                y_position_interp,
                                                                                aperture_radius,
                                                                                lens_plane_redshifts,
                                                                                delta_zs,
                                                                                include_mass_sheet_correction,
                                                                                rendering_geometry,
                                                                                global_render)

            if realization_type == 'composite_powerlaw':
                realization = realization_subs.join(realization_LOS, join_rendering_classes=True)

            elif realization_type == 'main_lens':
                realization = realization_subs

            else:
                realization = realization_LOS

            if realization_start is None:
                realization_start = realization
            else:
                realization_start = realization_start.join(realization)

            if verbose:
                new_foreground = realization.number_of_halos_before_redshift(self.zlens)
                new_background = realization.number_of_halos_after_redshift(self.zlens)
                new_subhalos = realization.number_of_halos_at_redshift(self.zlens)
                print('added ' + str(new_foreground) + ' halos before the lens redshift around coordinate '+str(j+1))
                print('added ' + str(new_subhalos) + ' halos at the lens redshift ' + str(j + 1))
                print('added ' + str(new_background) + ' halos after the lens redshift around coordinate '+str(j+1))

        return realization_start

    def _render_dynamic_main_single(self, arg_i, rendering_center_x, rendering_center_y, aperture_radius,
                                    x_centroid_main, y_centroid_main, include_mass_sheet_correction, geometry_render,
                                    render_index):

        args_render = deepcopy(arg_i)

        if self.zlens < args_render['zmin'] or self.zlens > args_render['zmax']:
            return None, None

        d = self.cosmology.D_C_z(self.zlens)
        x_center, y_center = rendering_center_x(d), rendering_center_y(d)

        # Make cuts on mass based on log_mlow/log_mhigh
        log_mlow_cut = args_render['log_mlow']
        log_mhigh_cut = args_render['log_mhigh']

        args_render['log_mlow'] = args_render['log_mlow_subs']
        args_render['log_mhigh'] = args_render['log_mhigh_subs']

        rendering_class_main = MainLensPowerLawDynamic(args_render, geometry_render, x_centroid_main, y_centroid_main)
        masses, x, y, r2d, r3d, redshifts = rendering_class_main(x_center, y_center, args_render['log_mlow'],
                                                                 args_render['log_mhigh'], aperture_radius)

        is_subhalo = True
        flag = [is_subhalo] * len(masses)
        mdefs = [args_render['mdef_main']]*len(masses)

        # mass sheet correction is False because we don't want to override the main mass sheet funciton
        # in the realization created with the largest LOS halos
        halo_mass_function = self.build_LOS_mass_function()
        realization = Realization(masses, x, y, r3d, mdefs, redshifts, flag, halo_mass_function, halo_profile_args=args_render,
                                  mass_sheet_correction=include_mass_sheet_correction, dynamic=True, rendering_classes=rendering_class_main,
                                  lens_cosmo_class=None, rendering_center_x=rendering_center_x, rendering_center_y=rendering_center_y)

        return realization, self._rendering_class_main[render_index]

    def _render_dynamic_los(self, args, x_position_interp, y_position_interp, aperture_radius,
                            lens_plane_redshifts, delta_zs, include_mass_sheet_correction, geometry_render,
                            global_render):

        realization = None
        rendering_class_return = False

        for count, args_i in enumerate(args):

            realization_i, rendering_class = self._render_dynamic_los_single(args_i, x_position_interp, y_position_interp, aperture_radius,
                            lens_plane_redshifts, delta_zs, include_mass_sheet_correction, geometry_render,
                            global_render)
            if count == 0:
                rendering_class_return = rendering_class
                realization = realization_i
            else:
                realization = realization_i.join(realization)

        assert rendering_class_return is not False

        return realization, rendering_class_return

    def _render_dynamic_los_single(self, args_i, x_position_interp, y_position_interp, aperture_radius,
                            lens_plane_redshifts, delta_zs, include_mass_sheet_correction, geometry_render,
                            global_render):

        args_render = deepcopy(args_i)

        if args_render['mass_func_type'] == 'POWER_LAW':

            rendering_class = LOSPowerLawDynamic(args_render, self.halo_mass_function, geometry_render,
                                                 aperture_radius, global_render,
                                                 lens_plane_redshifts, delta_zs)

            masses, x, y, r2d, r3d, redshifts = render_los_dynamic(rendering_class, aperture_radius,
                                                                   lens_plane_redshifts, delta_zs,
                                                                   x_position_interp, y_position_interp,
                                                                   args_render['zmin'], args_render['zmax'],
                                                                   self.geometry._cosmo.D_C_z)
            is_subhalo = False
            mdef = args_render['mdef_los']

        elif args_render['mass_func_type'] == 'DELTA':

            minimum_mass = args_render['log_mlow']
            rendering_class = LOSDeltaDynamic(args_render, self.halo_mass_function, geometry_render,
                                              aperture_radius, global_render,
                                              minimum_mass, lens_plane_redshifts, delta_zs)

            masses, x, y, r2d, r3d, redshifts = render_los_dynamic(rendering_class, aperture_radius,
                                                                   lens_plane_redshifts, delta_zs,
                                                                   x_position_interp, y_position_interp,
                                                                   args_render['zmin'], args_render['zmax'],
                                                                   self.geometry._cosmo.D_C_z)
            is_subhalo = False
            mdef = args_render['mdef_los']

        else:
            raise Exception('mass function type '+str(args_render['mass_func_type']) + ' not recognized.')

        flag = [is_subhalo] * len(masses)
        mdefs = [mdef] * len(masses)

        # mass sheet correction is False because we don't want to override the main mass sheet funciton
        # in the realization created with the largest LOS halos
        realization = Realization(masses, x, y, r2d, r3d, mdefs, redshifts, flag, self.halo_mass_function,
                                  halo_profile_args=args_render, mass_sheet_correction=include_mass_sheet_correction,
                                  dynamic=True)

        return realization, rendering_class

    @staticmethod
    def _lenstronomy_args_from_lensmodel(lensmodel):

        lens_model_list = lensmodel.lens_model_list
        redshift_list = lensmodel.redshift_list
        convention_index = lensmodel.lens_model._observed_convention_index
        return lens_model_list, redshift_list, convention_index

    @staticmethod
    def _interpolate_ray_paths(x_image, y_image, lens_model, kwargs_lens, zsource,
                              terminate_at_source=False, source_x=None, source_y=None):

        ray_angles_x = []
        ray_angles_y = []

        for (xi, yi) in zip(x_image, y_image):
            x, y, redshifts, tz = lens_model.lens_model.ray_shooting_partial_steps(0., 0., xi, yi, 0, zsource,
                                                                                   kwargs_lens)

            angle_x = [xi] + [x_comoving / tzi for x_comoving, tzi in zip(x[1:], tz[1:])]
            angle_y = [yi] + [y_comoving / tzi for y_comoving, tzi in zip(y[1:], tz[1:])]

            if terminate_at_source:
                assert source_x is not None
                assert source_y is not None
                angle_x[-1] = source_x
                angle_y[-1] = source_y

            ray_angles_x.append(interp1d(redshifts, angle_x))
            ray_angles_y.append(interp1d(redshifts, angle_y))

        return ray_angles_x, ray_angles_y









