from pyHalo.Rendering.Field.PowerLaw.powerlaw_dynamic import LOSPowerLawDynamic
from pyHalo.Rendering.Field.Delta.delta_dynamic import LOSDeltaDynamic
from pyHalo.Rendering.Main.mainlens_dynamic import MainLensPowerLawDynamic
from pyHalo.pyhalo_base import pyHaloBase
from pyHalo.single_realization import Realization
from pyHalo.Rendering.render import render_los_dynamic, render_main_dynamic
from pyHalo.Cosmology.geometry import Geometry
from scipy.interpolate import interp1d

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

        self._rendering_class_main = None
        super(pyHaloDynamic, self).__init__(zlens, zsource, cosmology_kwargs, kwargs_halo_mass_function)

    def reset(self, zlens, zsource):

        self._rendering_class_main = None
        self.reset_redshifts(zlens, zsource)

    @staticmethod
    def interpolate_ray_paths(x_coordinates, y_coordinates, lens_model, kwargs_lens, zsource,
                              terminate_at_source=False, source_x=None, source_y=None):

        """
        :param x_coordinates: x coordinates to interpolate (arcsec)
        :param y_coordinates: y coordinates to interpolate (arcsec)
        :param lens_model: instance of LensModel
        :param kwargs_lens: keyword arguments for lens model
        :param zsource: source redshift
        :param terminate_at_source: fix the final angular coordinate to the source coordinate
        :param source_x: source x coordinate (arcsec)
        :param source_y: source y coordinate (arcsec)
        :return: Instances of interp1d (scipy) that return the angular coordinate of a ray given a
        comoving distance
        """
        ray_angles_x = []
        ray_angles_y = []

        for (xi, yi) in zip(x_coordinates, y_coordinates):
            x, y, redshifts, tz = lens_model.lens_model.ray_shooting_partial_steps(0., 0., xi, yi, 0, zsource,
                                                                                   kwargs_lens)

            angle_x = [xi] + [x_comoving / tzi for x_comoving, tzi in zip(x[1:], tz[1:])]
            angle_y = [yi] + [y_comoving / tzi for y_comoving, tzi in zip(y[1:], tz[1:])]

            if terminate_at_source:
                angle_x[-1] = source_x
                angle_y[-1] = source_y

            ray_angles_x.append(interp1d(tz, angle_x))
            ray_angles_y.append(interp1d(tz, angle_y))

        return ray_angles_x, ray_angles_y

    def render_dynamic_with_macromodel(self, type, args, realization_start,
                     lens_centroid_x, lens_centroid_y,
                     macromodel_lensModel, kwargs_macromodel,
                     source_x, source_y, aperture_radius, verbose=False,
                       include_mass_sheet_correction=True):

        """

        :param type:
        :param args:
        :param realization_start:
        :param lens_centroid_x:
        :param lens_centroid_y:
        :param macromodel_lensModel:
        :param kwargs_macromodel:
        :param source_x:
        :param source_y:
        :param aperture_radius:
        :param verbose:
        :param include_mass_sheet_correction:
        :param global_render:
        :return:
        """

        x_interp_list, y_interp_list = self.interpolate_ray_paths([lens_centroid_x], [lens_centroid_y],
                                                                  macromodel_lensModel, kwargs_macromodel, self.zsource,
                                                                  terminate_at_source=True, source_x=source_x,
                                                                  source_y=source_y)

        return self.render_dynamic(type, args, realization_start, lens_centroid_x, lens_centroid_y, x_interp_list,
                                   y_interp_list, aperture_radius, verbose, include_mass_sheet_correction,
                                   global_render=True)

    def render_dynamic(self, type, args, realization_start, lens_centroid_x, lens_centroid_y,
                       x_interp_list, y_interp_list, aperture_radius,
                       verbose=False, include_mass_sheet_correction=False, global_render=False):

        """

        :param type:
        :param args:
        :param realization_start:
        :param lens_centroid_x:
        :param lens_centroid_y:
        :param x_interp_list:
        :param y_interp_list:
        :param aperture_radius:
        :param lens_plane_redshifts:
        :param delta_zs:
        :param verbose:
        :param include_mass_sheet_correction:
        :param global_render: if True, then the Geometry class is the one specified in kwargs_halo_mass function.
        If False, then a DOUBLE_CONE geometry is used. This is important because it affects how halos are
        rendered in apertures.

        Recommended usage is an initial `global' rendering with global_render = True, followed by a local render around
        each lensed image.
        :return:
        """

        lens_plane_redshifts, delta_zs = self.lens_plane_redshifts(args)

        self.halo_mass_function = self.build_LOS_mass_function(args)
        self.geometry = self.halo_mass_function.geometry

        if global_render is True:
            geometry_render = self.geometry

        else:
            geometry_render = Geometry(self.cosmology, self.zlens, self.zsource,
                                       self.geometry.cone_opening_angle, 'DOUBLE_CONE')

            if include_mass_sheet_correction:
                print('WARNING: You specified include_mass_sheet_corretion = True for a local rendering of halos,'
                      'you should probably only do this for a global rendering of halos throughout the entire volume.')

        kwargs_init = self._add_profile_params(args, True)

        args = deepcopy(kwargs_init)

        rendering_class_main, rendering_class_LOS = None, None

        for j, (x_position_interp, y_position_interp) in enumerate(zip(x_interp_list, y_interp_list)):

            realization_new, rendering_class_main, rendering_class_LOS = self._render(type, args, aperture_radius,
                                           lens_centroid_x, lens_centroid_y,
                                           x_position_interp, y_position_interp,
                                           lens_plane_redshifts, delta_zs, include_mass_sheet_correction, geometry_render)

            if realization_start is None:
                realization_start = realization_new
            elif realization_new is not None:
                realization_start = realization_start.join(realization_new)

            if verbose:

                new_foreground = realization_new.number_of_halos_before_redshift(self.zlens)
                new_background = realization_new.number_of_halos_after_redshift(self.zlens)
                new_subhalos = realization_new.number_of_halos_at_redshift(self.zlens)
                print('added ' + str(new_foreground) + ' foreground halos around coordinate '+str(j+1))
                print('added ' + str(new_subhalos) + ' subhalos halos around coordinate ' + str(j + 1))
                print('added ' + str(new_background) + ' background halos around coordinate '+str(j+1))

        rendering_classes = []

        if rendering_class_main is not None:
            rendering_classes += [rendering_class_main]
        if rendering_class_LOS is not None:
            rendering_classes += [rendering_class_LOS]

        realization_start.set_rendering_classes(rendering_classes)

        return realization_start

    def _render(self, type, args, aperture_radius, x_centroid_main, y_centroid_main,
                            x_position_interp, y_position_interp,
                            lens_plane_redshifts, delta_zs, include_mass_sheet_correction, geometry_render):

        rendering_class_main, rendering_class_LOS = None, None

        if type == 'main_lens' or type == 'composite_powerlaw':

            realization, rendering_class_main = self._render_dynamic_main(args, x_position_interp, y_position_interp,
                                                    aperture_radius, x_centroid_main, y_centroid_main,
                                                    include_mass_sheet_correction, geometry_render)

            if type == 'composite_powerlaw':

                realization_LOS, rendering_class_LOS = self._render_dynamic_los(args, x_position_interp, y_position_interp,
                                                           aperture_radius, lens_plane_redshifts, delta_zs,
                                                           include_mass_sheet_correction, geometry_render)

                if realization is None:
                    realization = realization_LOS
                else:
                    realization = realization.join(realization_LOS)

        elif type == 'line_of_sight':

            realization, rendering_class_LOS = self._render_dynamic_los(args, x_position_interp, y_position_interp,
                                                       aperture_radius, lens_plane_redshifts, delta_zs,
                                                   include_mass_sheet_correction, geometry_render)

        else:
            raise Exception('rendering type '+str(type) + ' not recognized.')

        return realization, rendering_class_main, rendering_class_LOS

    def _render_dynamic_main(self, args, x_aperture, y_aperture, aperture_radius,
                             x_centroid_main, y_centroid_main, include_mass_sheet_correction, geometry_render):

        args_render = deepcopy(args)

        if self.zlens < args_render['zmin'] or self.zlens > args_render['zmax']:
            return None, None

        x_window_location, y_window_location = x_aperture(self.zlens), y_aperture(self.zlens)

        # Make cuts on mass based on log_mlow/log_mhigh
        log_mlow_cut = args_render['log_mlow']
        log_mhigh_cut = args_render['log_mhigh']

        if self._rendering_class_main is None:

            # Render subhalos include all masses, make cuts on mass and position later
            args_render['log_mlow'] = args_render['log_mlow_subs']
            args_render['log_mhigh'] = args_render['log_mhigh_subs']

            self._rendering_class_main = MainLensPowerLawDynamic(args_render, geometry_render,
                                                                 x_centroid_main, y_centroid_main)

            #print('nhalos: ',len(self._rendering_class_main._masses))

        masses, x, y, r2d, r3d, redshifts = render_main_dynamic(self._rendering_class_main, aperture_radius,
                                                        x_window_location, y_window_location,
                                                        log_mlow_cut, log_mhigh_cut)



        is_subhalo = True
        flag = [is_subhalo] * len(masses)
        mdefs = [args_render['mdef_main']]*len(masses)

        # mass sheet correction is False because we don't want to override the main mass sheet funciton
        # in the realization created with the largest LOS halos
        realization = Realization(masses, x, y, r2d, r3d, mdefs, redshifts, flag, self.halo_mass_function,
                                  other_params=args_render, mass_sheet_correction=include_mass_sheet_correction,
                                  dynamic=True)

        return realization, self._rendering_class_main

    def _render_dynamic_los(self, args, x_position_interp, y_position_interp, aperture_radius,
                            lens_plane_redshifts, delta_zs, include_mass_sheet_correction, geometry_render):

        args_render = deepcopy(args)

        if args['mass_func_type'] == 'POWER_LAW':

            rendering_class = LOSPowerLawDynamic(args_render, self.halo_mass_function, geometry_render,
                                                 aperture_radius,
                                                 lens_plane_redshifts, delta_zs)

            masses, x, y, r2d, r3d, redshifts = render_los_dynamic(rendering_class, aperture_radius,
                                                                   lens_plane_redshifts, delta_zs,
                                                                   x_position_interp, y_position_interp,
                                                                   args['zmin'], args['zmax'],
                                                                   self.geometry._cosmo.D_C_z)
            is_subhalo = False
            mdef = args_render['mdef_los']

        elif args['mass_func_type'] == 'DELTA':

            minimum_mass = args_render['log_mlow']
            rendering_class = LOSDeltaDynamic(args_render, self.halo_mass_function, geometry_render,
                                              aperture_radius, minimum_mass,
                                                 lens_plane_redshifts, delta_zs)

            masses, x, y, r2d, r3d, redshifts = render_los_dynamic(rendering_class, aperture_radius,
                                                                   lens_plane_redshifts, delta_zs,
                                                                   x_position_interp, y_position_interp,
                                                                   args['zmin'], args['zmax'],
                                                                   self.geometry._cosmo.D_C_z)
            is_subhalo = False
            mdef = args_render['mdef_los']

        else:
            raise Exception('mass function type '+str(args['mass_func_type']) + ' not recognized.')

        flag = [is_subhalo] * len(masses)
        mdefs = [mdef] * len(masses)

        # mass sheet correction is False because we don't want to override the main mass sheet funciton
        # in the realization created with the largest LOS halos
        realization = Realization(masses, x, y, r2d, r3d, mdefs, redshifts, flag, self.halo_mass_function,
                                  other_params=args_render, mass_sheet_correction=include_mass_sheet_correction,
                                  dynamic=True)

        return realization, rendering_class

    def interpolated_ray_paths(self, lens_model, kwargs, x_coords, y_coords,
                    source_x, source_y, lens_plane_redshifts):

        from lenstronomy.LensModel.lens_model import LensModel

        lens_model_list_empty = []
        zlist_empty = []
        kwargs_empty = []
        for i, zi in enumerate(lens_plane_redshifts):
            lens_model_list_empty.append('CONVERGENCE')
            zlist_empty.append(zi)
            kwargs_empty.append({'kappa_ext': 0.})

        lens_model_list, redshift_list, convention_index = \
            self._lenstronomy_args_from_lensmodel(lens_model)

        lens_model_list += lens_model_list_empty
        redshift_list += zlist_empty
        kwargs += kwargs_empty

        lensmodel_new = LensModel(lens_model_list, z_lens=self.zlens,
                                  z_source=self.zsource, lens_redshift_list=redshift_list,
                                  cosmo=lens_model.cosmo, multi_plane=True,
                                  numerical_alpha_class=None)

        x_interp, y_interp = self._interpolate_ray_paths(x_coords, y_coords,
                              lensmodel_new, kwargs, self.zsource, terminate_at_source=True,
                                                   source_x=source_x, source_y=source_y)

        return x_interp, y_interp

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









