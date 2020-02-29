from pyHalo.Rendering.Field.field_dynamic import LOSPowerLawDynamic
from pyHalo.Rendering.Main.mainlens_dynamic import MainLensPowerLawDynamic
from pyHalo.pyhalo import pyHalo
from pyHalo.single_realization import Realization

from pyHalo.defaults import *

from copy import deepcopy

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel

class pyHaloDynamic(pyHalo):

    def __init__(self, log_mlow_global, macromodel_lensmodel, kwargs_lens_macro, zlens, zsource, cosmology_kwargs={},
                 kwargs_halo_mass_function={}, kwargs_render={}):

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

        self._macro_lens_model = macromodel_lensmodel
        self._macro_kwargs = kwargs_lens_macro
        self._lensmodel_setup, self._kwargs_setup, self._lens_plane_redshifts, self._delta_zs = \
            self._setup(zlens, zsource, macromodel_lensmodel, kwargs_lens_macro, kwargs_render)
        self._log_mlow_global = log_mlow_global

        self._x_center_lens, self._y_center_lens = self._lens_cone_center_lens(
            self._lensmodel_setup, self._kwargs_setup, 0., 0.)

        super(pyHaloDynamic, self).__init__(zlens, zsource, cosmology_kwargs, kwargs_halo_mass_function)

    def render(self, type, args, nrealizations=1, verbose=False):

        realizations = []

        for n in range(nrealizations):

            args = self._add_profile_params(args)
            args['log_mlow'] = self._log_mlow_global
            realizations.append(self._render_single(type, args, verbose))

        return realizations

    def render_dynamic(self, type, args, x_angle, y_angle, rmax_2d,
                       log_mlow, log_mhigh, realization_global, verbose=False):

        lens_model_list_global, redshift_list_global, kwargs_lens_global, numerical_alpha_class = \
            realization_global.lensing_quantities(mass_sheet_front=self._log_mlow_global,
                                                  mass_sheet_correction_back=self._log_mlow_global)

        lens_list_macro, zlist_macro, convention_idx_macro = self._lenstronomy_args_from_lensmodel(self._macro_lens_model)

        lens_model_list = lens_list_macro + lens_model_list_global
        redshift_list = zlist_macro + realization_global
        kwargs_lens = self._macro_kwargs + kwargs_lens_global

        lensmodel_global = LensModel(lens_model_list, lens_redshift_list=redshift_list,
                                     cosmo=self._lensmodel_setup.cosmo, multi_plane=True,
                                     observed_convention_index=convention_idx_macro,
                                     numerical_alpha_class=numerical_alpha_class)

        if type == 'main_lens' or type == 'composite_powerlaw':

            x_aperture_position, y_aperture_position = self._lens_cone_center_lens(
                lensmodel_global, kwargs_lens, x_angle, y_angle)

            realization_subhalos = self._render_single('dynamic_main', args, verbose, log_mlow, log_mhigh,
                                                       x_aperture_position, y_aperture_position, rmax_2d)

            realization_global = realization_subhalos.update(realization_global)

        if type == 'composite_powerlaw':

            x_aperture_position, y_aperture_position = self._lens_cone_center(
                lensmodel_global, kwargs_lens, x_angle, y_angle)

            realization_los = self._render_single('dynamic_LOS', args, verbose, log_mlow, log_mhigh,
                                                       x_aperture_position, y_aperture_position, rmax_2d)

            realization_global = realization_los.update(realization_global)

        return realization_global

    def _render_single_dynamic(self, type, args, verbose, log_mlow, log_mhigh,
                       x_aperture, y_aperture, aperture_size):

        if len(args) > 1:
            raise Exception('This class only handles single populations.')

        assert type in ['dynamic_main', 'dynamic_LOS']

        args_render = deepcopy(args)

        if log_mlow is None:
            args_render['log_mlow'] = self._log_mlow_global
        else:
            args_render['log_mlow'] = log_mlow

        if log_mhigh is not None:
            args_render['log_mhigh'] = log_mhigh

        self.halo_mass_function = self._build_LOS_mass_function(args_render)
        self._geometry = self.halo_mass_function.geometry

        flag, mdefs = [], []

        assert x_aperture is not None
        assert y_aperture is not None
        assert aperture_size is not None

        mass_sheet = False

        if type == 'dynamic_main':

            rendering_class = MainLensPowerLawDynamic(args, self._geometry, self._x_center_lens, self._y_center_lens,
                                                      x_aperture, y_aperture, aperture_size)

            masses, x, y, r2d, r3d, redshifts = rendering_class()
            mdef = args['mdef_main']
            flag = [True] * len(masses)
            mdefs += [mdef] * len(masses)

        elif type == 'dynamic_LOS':

            rendering_class = LOSPowerLawDynamic(args, self.halo_mass_function, self._lens_plane_redshifts,
                                                 self._delta_zs,
                                                 x_aperture, y_aperture, aperture_size)

            masses, x, y, r2d, r3d, redshifts = rendering_class()
            mdef = args['mdef_los']
            flag = [True] * len(masses)
            mdefs += [mdef] * len(masses)

        realization = Realization(masses, x, y, r2d, r3d, mdefs, redshifts, flag, self.halo_mass_function,
                                  other_params=args_render, mass_sheet_correction=mass_sheet)

        return realization

    @staticmethod
    def _lenstronomy_args_from_lensmodel(lensmodel):

        lens_model_list = lensmodel.lens_model_list
        redshift_list = lensmodel.lens_redshift_list
        convention_index = lensmodel.lens_model._observed_convention_index
        return lens_model_list, redshift_list, convention_index

    def _setup(self, zlens, zsource, lensmodel, kwargs_lens_macro, kwargs_render):

        redshifts, _ = self.lens_plane_redshifts(kwargs_render)
        models = ['CONVERGENCE'] * len(redshifts)
        kwargs_lens_planes = [{'kappa_ext': 0.}]*len(redshifts)

        lens_list_macro, zlist_macro, convention_idx_macro = self._lenstronomy_args_from_lensmodel(lensmodel)
        lens_model_list = lens_list_macro + models
        redshift_list = zlist_macro + redshifts
        convention_index = convention_idx_macro
        kwargs = kwargs_lens_macro + kwargs_lens_planes
        lensmodel_setup = LensModel(lens_model_list, z_lens=zlens, z_source=zsource, lens_redshift_list=redshift_list,
                                    observed_convention_index=convention_index, multi_plane=True, cosmo=lensmodel.cosmo)

        delta_zs = []
        for i in range(0, len(redshift_list)-1):
            delta_zs.append(redshift_list[i+1] - redshift_list[i])
        delta_zs.append(zsource - redshift_list[-1])

        return lensmodel_setup, kwargs, redshift_list, delta_zs

    def _lens_cone_center_lens(self, lensModel, kwargs_lens, alpha_x=0., alpha_y = 0.):

        x0, y0 = 0., 0.
        z_start = 0.
        z_stop = lensModel.z_lens

        comoving_x, comoving_y, _, Tz_list = lensModel.lens_model.ray_shooting_partial_steps(x0, y0,
                                                                                                   alpha_x, alpha_y,
                                                                                                   z_start, z_stop,
                                                                                                   kwargs_lens)

        return comoving_x[-1]/Tz_list[-1], comoving_y[-1]/Tz_list[-1]

    def _lens_cone_center(self, lensModel, kwargs_lens, alpha_x=0., alpha_y=0.):

        x0, y0 = 0., 0.
        z_start = 0.
        z_stop = lensModel.z_source

        comoving_x, comoving_y, _, Tz_list = lensModel.lens_model.ray_shooting_partial_steps(x0, y0,
                                     alpha_x, alpha_y, z_start, z_stop, kwargs_lens)

        angle_x, angle_y = [0.], [0.]
        for i in range(1, len(comoving_x)):
            angle_x.append(comoving_x[i] / Tz_list[i])
            angle_x.append(comoving_y[i] / Tz_list[i])
        angle_x, angle_y = np.array(angle_x), np.array(angle_y)

        return angle_x, angle_y










