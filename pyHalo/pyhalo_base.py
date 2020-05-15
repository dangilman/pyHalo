from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.defaults import *
import numpy as np
from scipy.interpolate import interp1d


class pyHaloBase(object):

    def __init__(self, zlens, zsource, cosmology_kwargs,
                 kwargs_halo_mass_function):

        """

        :param zlens: lens redshift
        :param zsource: source redshift
        :param cosmology_kwargs:
        keyword arguments for 'Cosmology' class. See documentation in cosmology.py
        :param halo_mass_function_args:
        keyword arguments for 'LensingMassFunction' class. See documentation in lensing_mass_function.py
        :param kwargs_massfunc:
        keyword arguments
        """

        self._cosmology_kwargs = cosmology_kwargs
        self._kwargs_mass_function = kwargs_halo_mass_function
        self._halo_mass_function_args = kwargs_halo_mass_function
        self.reset_redshifts(zlens, zsource)

    @staticmethod
    def interpolate_ray_paths(x_coordinates, y_coordinates, lens_model, kwargs_lens, zsource,
                              terminate_at_source=False, source_x=None, source_y=None):

        """
        :param x_coordinates: x coordinates to interpolate (arcsec) (list)
        :param y_coordinates: y coordinates to interpolate (arcsec) (list)
        Typically x_coordinates/y_coordinates would be four image positions, or the coordinate of the lens centroid
        :param lens_model: instance of LensModel (lenstronomy)
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

    def lens_plane_redshifts(self, kwargs_render=dict):

        zmin = lenscone_default.default_zstart
        if 'zstep' not in kwargs_render.keys():
            zstep = lenscone_default.default_z_step
        else:
            zstep = kwargs_render['zstep']
        zmax = self.zsource - zstep

        front_z = np.arange(zmin, self.zlens, zstep)
        back_z = np.arange(self.zlens, zmax, zstep)
        redshifts = np.append(front_z, back_z)

        delta_zs = []
        for i in range(0, len(redshifts) - 1):
            delta_zs.append(redshifts[i + 1] - redshifts[i])
        delta_zs.append(self.zsource - redshifts[-1])

        return list(np.round(redshifts, 2)), np.round(delta_zs, 2)

    def reset_redshifts(self, zlens, zsource):

        self.zlens = zlens
        self.zsource = zsource
        self.cosmology = Cosmology(**self._cosmology_kwargs)
        self.halo_mass_function = None
        self.geometry = None

    @property
    def astropy_cosmo(self):
        return self.cosmology.astropy

    def build_LOS_mass_function(self, args):

        if self.halo_mass_function is None:

            if 'mass_func_type' not in args.keys():
                args['mass_func_type'] = realization_default.default_type

            if args['mass_func_type'] == 'delta':

                logLOS_mlow = args['logM_delta'] - 0.01
                logLOS_mhigh = args['logM_delta'] + 0.01

            else:
                if 'log_mlow_los' not in args.keys():
                    logLOS_mlow = realization_default.log_mlow
                else:
                    logLOS_mlow = args['log_mlow_los']

                if 'log_mhigh_los' not in args.keys():
                    logLOS_mhigh = realization_default.log_mhigh
                else:
                    logLOS_mhigh = args['log_mhigh_los']

            if 'two_halo_term' not in self._halo_mass_function_args.keys():
                self._halo_mass_function_args.update({'two_halo_term': realization_default.two_halo_term})
            if 'mass_function_model' not in self._halo_mass_function_args.keys():
                self._halo_mass_function_args.update({'mass_function_model': cosmo_default.default_mass_function})

            self.halo_mass_function = LensingMassFunction(self.cosmology, 10 ** logLOS_mlow, 10 ** logLOS_mhigh, self.zlens, self.zsource,
                                                          cone_opening_angle=args['cone_opening_angle'],
                                                          **self._halo_mass_function_args)

        return self.halo_mass_function

    def _add_profile_params(self, args, dynamic):

        return set_default_kwargs(args, dynamic, self.zsource)






