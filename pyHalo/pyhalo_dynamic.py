from pyHalo.Rendering.Field.field_dynamic import LOSPowerLawDynamic
from pyHalo.Rendering.Field.delta_dynamic import LOSDeltaDynamic
from pyHalo.Rendering.Main.mainlens_dynamic import MainLensPowerLawDynamic
from pyHalo.pyhalo import pyHalo
from pyHalo.single_realization import Realization
from pyHalo.Rendering.render import render

from pyHalo.defaults import *

from copy import deepcopy

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel

class pyHaloDynamic(pyHalo):

    def __init__(self, zlens, zsource, cosmology_kwargs={},
                 kwargs_halo_mass_function={}):

        """
        :macromodel_lensmodel: instance of LensModel that containts just the macromodel (i.e. the big deflectors)
        :param zlens: lens redshift
        :param zsource: source redshift
        :param cosmology_kwargs:
        keyword arguments for 'Cosmology' class. See documentation in cosmology.py
        :param halo_mass_function_args:
        keyword arguments for 'LensingMassFunction' class. See documentation in lensing_mass_function.py
        :param kwargs_massfunc:
        keyword arguments
        """
        self._initialized = False
        super(pyHaloDynamic, self).__init__(zlens, zsource, cosmology_kwargs, kwargs_halo_mass_function)

    def set_macro_lensmodel(self, macromodel_lensmodel, kwargs_lens_macro, log_mlow_global, kwargs_render={}):

        self._macro_lens_model = macromodel_lensmodel
        self._macro_kwargs = kwargs_lens_macro

        self._lensmodel_setup, self._kwargs_setup, self._lens_plane_redshifts, self._delta_zs = \
            self._setup(self.zlens, self.zsource, macromodel_lensmodel, kwargs_lens_macro, kwargs_render)

        self._log_mlow_global = log_mlow_global
        self._x_center_lens, self._y_center_lens = self._lens_cone_center_lens(
            self._lensmodel_setup, self._kwargs_setup, 0., 0.)

        self._initialized = True

    def render(self, type, args, nrealizations=1, verbose=False):

        if not self._initialized:
            raise Exception('must initialize this class by passing in an instance of '
                            'LensModel that includes the macromodel. (see set_macro_lensmodel routine)')
        realizations = []

        for n in range(nrealizations):

            args = self._add_profile_params(args, dynamic=True)
            args['log_mlow'] = self._log_mlow_global
            realizations.append(self._render_single(type, args, verbose))

        return realizations

    def render_dynamic(self, type, args, x_angles, y_angles, rmax_2d,
                       log_mlow, log_mhigh, macro_lens_model, kwargs_macro, realization_global, verbose=False):

        if not self._initialized:
            raise Exception('must initialize this class by passing in an instance of '
                            'LensModel that includes the macromodel. (see set_macro_lensmodel routine)')

        args = self._add_profile_params(args, True)

        for j, (x_angle, y_angle) in enumerate(zip(x_angles, y_angles)):
            realization = self._render_in_aperture(type, args, x_angle, y_angle, rmax_2d,
                                              log_mlow, log_mhigh, macro_lens_model, kwargs_macro, realization_global, verbose)

            if realization is not None:
                realization_global = realization_global.join(realization)

            if verbose:
                print('x_coordinate: ', x_angle)
                print('y_coordinate: ', y_angle)
                new_foreground = realization.number_of_halos_before_redshift(self.zlens)
                new_background = realization.number_of_halos_after_redshift(self.zlens)
                print('added ' + str(new_foreground) + ' foreground halos around image '+str(j+1))
                print('added ' + str(new_background) + ' background halos around image '+str(j+1))

        return realization_global

    def _render_in_aperture(self, type, args, x_angle, y_angle, rmax_2d,
                       log_mlow, log_mhigh, macro_lens_model, kwargs_macro, realization_global, verbose):

        lens_model_list_global, redshift_list_global, kwargs_lens_global, numerical_alpha_class = \
            realization_global.lensing_quantities(mass_sheet_correction_front=self._log_mlow_global,
                                                  mass_sheet_correction_back=self._log_mlow_global)

        lens_list_macro, zlist_macro, convention_idx_macro = \
            self._lenstronomy_args_from_lensmodel(macro_lens_model)

        lens_model_list = lens_list_macro + lens_model_list_global
        redshift_list = zlist_macro + list(np.round(redshift_list_global, 2))
        kwargs_lens = kwargs_macro + kwargs_lens_global

        lensmodel_global = LensModel(lens_model_list, lens_redshift_list=redshift_list,
                                     cosmo=self._lensmodel_setup.cosmo, multi_plane=True,
                                     z_source=self.zsource,
                                     observed_convention_index=convention_idx_macro,
                                     numerical_alpha_class=numerical_alpha_class)

        if type == 'main_lens' or type == 'composite_powerlaw':

            x_aperture_position, y_aperture_position = self._lens_cone_center_lens(
                lensmodel_global, kwargs_lens, x_angle, y_angle)

            realization = self._render_single_dynamic('dynamic_main', args, verbose, log_mlow, log_mhigh,
                                                       x_aperture_position, y_aperture_position, rmax_2d)

            if type == 'composite_powerlaw':

                x_aperture, y_aperture = self._lens_cone_center(lens_model_list, redshift_list, kwargs_lens, x_angle,
                                                                y_angle, numerical_alpha_class)
                realization = self._render_single_dynamic('dynamic_LOS', args, verbose, log_mlow, log_mhigh,
                                                           x_aperture, y_aperture, rmax_2d)

        elif type == 'line_of_sight':

            x_aperture, y_aperture = self._lens_cone_center(lens_model_list, redshift_list, kwargs_lens, x_angle,
                                                            y_angle, numerical_alpha_class)
            realization = self._render_single_dynamic('dynamic_LOS', args, verbose, log_mlow, log_mhigh,
                                                      x_aperture, y_aperture, rmax_2d)

        return realization

    def _render_single_dynamic(self, type, args, verbose, log_mlow, log_mhigh,
                       x_aperture, y_aperture, aperture_size):

        assert type in ['dynamic_main', 'dynamic_LOS']

        args_render = deepcopy(args)
        args_render['log_mlow'] = log_mlow
        args_render['log_mhigh'] = log_mhigh

        flag, mdefs = [], []

        assert x_aperture is not None
        assert y_aperture is not None
        assert aperture_size is not None

        mass_sheet = False

        lens_plane_redshifts, delta_zs = list(np.round(self._lens_plane_redshifts, 2)), self._delta_zs,

        if type == 'dynamic_main':

            if self.zlens < args_render['zmin'] or self.zlens > args_render['zmax']:
                return None

            rendering_class = MainLensPowerLawDynamic(args_render, self._geometry, self._x_center_lens, self._y_center_lens,
                                                      x_aperture, y_aperture, aperture_size)

            masses, x, y, r2d, r3d, redshifts = render(rendering_class)
            is_subhalo = True
            mdef = args_render['mdef_main']

        elif type == 'dynamic_LOS_powerlaw':

            rendering_class = LOSPowerLawDynamic(args_render, self.halo_mass_function,
                                                 x_aperture, y_aperture, aperture_size)

            masses, x, y, r2d, r3d, redshifts = render(rendering_class, lens_plane_redshifts, delta_zs)
            is_subhalo = False
            mdef = args_render['mdef_los']

        elif type == 'dynamic_LOS_delta':

            rendering_class = LOSDeltaDynamic(args_render, self.halo_mass_function,
                                                 x_aperture, y_aperture, aperture_size)

            masses, x, y, r2d, r3d, redshifts = render(rendering_class, lens_plane_redshifts, delta_zs)
            is_subhalo = False
            mdef = args_render['mdef_los']

        else:
            raise Exception('realization type '+str(type)+ ' not recognized.')

        flag = [is_subhalo] * len(masses)
        mdefs += [mdef] * len(masses)

        realization = Realization(masses, x, y, r2d, r3d, mdefs, redshifts, flag, self.halo_mass_function,
                                  other_params=args_render, mass_sheet_correction=mass_sheet, dynamic=True)

        return realization

    @staticmethod
    def _lenstronomy_args_from_lensmodel(lensmodel):

        lens_model_list = lensmodel.lens_model_list
        redshift_list = lensmodel.redshift_list
        convention_index = lensmodel.lens_model._observed_convention_index
        return lens_model_list, redshift_list, convention_index

    def _setup(self, zlens, zsource, lensmodel, kwargs_lens_macro, kwargs_render):

        redshifts, delta_zs = self.lens_plane_redshifts(kwargs_render)
        redshifts = list(np.round(redshifts, 2))

        models = ['CONVERGENCE'] * len(redshifts)
        kwargs_lens_planes = [{'kappa_ext': 0.}]*len(redshifts)

        lens_list_macro, zlist_macro, convention_idx_macro = self._lenstronomy_args_from_lensmodel(lensmodel)
        lens_model_list = lens_list_macro + models
        redshift_list = zlist_macro + list(redshifts)
        convention_index = convention_idx_macro
        kwargs = kwargs_lens_macro + kwargs_lens_planes
        lensmodel_setup = LensModel(lens_model_list, z_lens=zlens, z_source=zsource, lens_redshift_list=redshift_list,
                                    observed_convention_index=convention_index, multi_plane=True, cosmo=lensmodel.cosmo)

        return lensmodel_setup, kwargs, list(np.round(redshifts, 2)), np.round(delta_zs, 2)

    def _lens_cone_center_lens(self, lensModel, kwargs_lens, alpha_x=0., alpha_y = 0.):

        x0, y0 = 0., 0.
        z_start = 0.
        z_stop = self.zlens

        comoving_x, comoving_y, _, Tz_list = lensModel.lens_model.ray_shooting_partial_steps(x0, y0, alpha_x,
                                                                                             alpha_y, z_start, z_stop,
                                                                                             kwargs_lens)

        return comoving_x[-1]/Tz_list[-1], comoving_y[-1]/Tz_list[-1]

    def _lens_cone_center(self, lens_model_list, redshift_list, kwargs_lens, alpha_x=0., alpha_y=0., numerical_alpha_class=None):

        x0, y0 = 0., 0.
        z_start = 0.
        z_stop = self.zsource

        redshift_list_full = list(np.round(redshift_list,2)) + list(np.round(self._lens_plane_redshifts,2))
        lens_model_list_full = lens_model_list + ['CONVERGENCE']*len(self._lens_plane_redshifts)
        kwargs_full = kwargs_lens + [{'kappa_ext': 0.}] * len(self._lens_plane_redshifts)

        lensModel = LensModel(lens_model_list_full, z_lens=self.zlens, z_source=self.zsource,
                              lens_redshift_list=redshift_list_full, cosmo=self._lensmodel_setup.cosmo,
                              multi_plane=True, numerical_alpha_class=numerical_alpha_class)

        comoving_x, comoving_y, _, Tz_list = lensModel.lens_model.ray_shooting_partial_steps(x0, y0,
                                     alpha_x, alpha_y, z_start, z_stop, kwargs_full)

        angle_x, angle_y = [], []
        for i in range(1, len(comoving_x)-1):

            angle_x.append(comoving_x[i] / Tz_list[i])
            angle_y.append(comoving_y[i] / Tz_list[i])

        angle_x, angle_y = np.array(angle_x), np.array(angle_y)

        return angle_x, angle_y










