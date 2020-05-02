from pyHalo.Rendering.Field.field_dynamic import LOSPowerLawDynamic
from pyHalo.Rendering.Field.delta_dynamic import LOSDeltaDynamic
from pyHalo.Rendering.Main.mainlens_dynamic import MainLensPowerLawDynamic
from pyHalo.pyhalo_base import pyHaloBase
from pyHalo.single_realization import Realization
from pyHalo.Rendering.render import render_los_dynamic, render_main_dynamic

from pyHalo.defaults import *

from copy import deepcopy

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel

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

    def render_dynamic(self, type, args, realization_start, lens_centroid_x, lens_centroid_y,
                       x_interp_list, y_interp_list, aperture_radius,
                       lens_plane_redshifts, delta_zs, verbose=False,
                       include_mass_sheet_correction=False):

        self.halo_mass_function = self.build_LOS_mass_function(args)
        self._geometry = self.halo_mass_function.geometry

        args = self._add_profile_params(args, True)

        # args['log_mlow'] = log_mlow
        # args['log_mhigh'] = log_mhigh

        for j, (x_position_interp, y_position_interp) in enumerate(zip(x_interp_list, y_interp_list)):

            realization_new = self._render(type, args, aperture_radius,
                                           lens_centroid_x, lens_centroid_y,
                                           x_position_interp, y_position_interp,
                                           lens_plane_redshifts, delta_zs, include_mass_sheet_correction)

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

        return realization_start

    def _render(self, type, args, aperture_radius, x_centroid_main, y_centroid_main,
                            x_position_interp, y_position_interp,
                            lens_plane_redshifts, delta_zs, include_mass_sheet_correction):

        if type == 'main_lens' or type == 'composite_powerlaw':

            realization = self._render_dynamic_main(args, x_position_interp, y_position_interp,
                                                    aperture_radius, x_centroid_main, y_centroid_main,
                                                    include_mass_sheet_correction)

            if type == 'composite_powerlaw':

                realization_LOS = self._render_dynamic_los(args, x_position_interp, y_position_interp,
                                                           aperture_radius, lens_plane_redshifts, delta_zs,
                                                           include_mass_sheet_correction)

                if realization is None:
                    realization = realization_LOS
                else:
                    realization = realization.join(realization_LOS)

        elif type == 'line_of_sight':

            realization = self._render_dynamic_los(args, x_position_interp, y_position_interp,
                                                       aperture_radius, lens_plane_redshifts, delta_zs,
                                                   include_mass_sheet_correction)

        else:
            raise Exception('rendering type '+str(type) + ' not recognized.')

        return realization

    def _render_dynamic_main(self, args, x_aperture, y_aperture, aperture_radius,
                             x_centroid_main, y_centroid_main, include_mass_sheet_correction):

        args_render = deepcopy(args)

        if self.zlens < args_render['zmin'] or self.zlens > args_render['zmax']:
            return None

        x_window_location, y_window_location = x_aperture(self.zlens), y_aperture(self.zlens)

        # Make cuts on mass based on log_mlow/log_mhigh
        log_mlow_cut = args_render['log_mlow']
        log_mhigh_cut = args_render['log_mhigh']

        if self._rendering_class_main is None:

            # Render subhalos include all masses, make cuts on mass and position later
            args_render['log_mlow'] = args_render['log_mlow_subs']
            args_render['log_mhigh'] = args_render['log_mhigh_subs']

            self._rendering_class_main = MainLensPowerLawDynamic(args_render, self._geometry,
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

        return realization

    def _render_dynamic_los(self, args, x_position_interp, y_position_interp, aperture_radius,
                            lens_plane_redshifts, delta_zs, include_mass_sheet_correction):

        args_render = deepcopy(args)

        if args['mass_func_type'] == 'POWER_LAW':

            rendering_class = LOSPowerLawDynamic(args_render, self.halo_mass_function, aperture_radius)

            masses, x, y, r2d, r3d, redshifts = render_los_dynamic(rendering_class, aperture_radius,
                                                                   lens_plane_redshifts, delta_zs,
                                                                   x_position_interp, y_position_interp,
                                                                   args['zmin'], args['zmax'])
            is_subhalo = False
            mdef = args_render['mdef_los']

        elif args['mass_func_type'] == 'DELTA':

            minimum_mass = args_render['log_mlow']
            rendering_class = LOSDeltaDynamic(args_render, self.halo_mass_function, aperture_radius, minimum_mass)

            masses, x, y, r2d, r3d, redshifts = render_los_dynamic(rendering_class, aperture_radius,
                                                                   lens_plane_redshifts, delta_zs,
                                                                   x_position_interp, y_position_interp,
                                                                   args['zmin'], args['zmax'])
            is_subhalo = False
            mdef = args_render['mdef_los']

        flag = [is_subhalo] * len(masses)
        mdefs = [mdef] * len(masses)

        # mass sheet correction is False because we don't want to override the main mass sheet funciton
        # in the realization created with the largest LOS halos
        realization = Realization(masses, x, y, r2d, r3d, mdefs, redshifts, flag, self.halo_mass_function,
                                  other_params=args_render, mass_sheet_correction=include_mass_sheet_correction,
                                  dynamic=True)

        return realization









