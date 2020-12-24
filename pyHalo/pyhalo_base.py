from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
import numpy as np
from pyHalo.defaults import *
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

    def interpolate_ray_paths(self, x_coordinates, y_coordinates, lens_model, kwargs_lens, zsource,
                               terminate_at_source=False, source_x=None, source_y=None, evaluate_at_mean=False):

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
        :param evaluate_at_mean: if True, returns two single interp1d instances (one for each of x/y) that return the
        average of each individual x/y coordinate evaluated at each lens plane. For example, if you pass in four images positions
        the output would be an interpolation of the average x/y coordinate along the path traversed by the light
        (This is useful for aligning realizations with a background source significantly offset from the lens centroid)

        :return: Instances of interp1d (scipy) that return the angular coordinate of a ray given a
        comoving distance
        """

        angle_x = []
        angle_y = []

        for i, (xpos, ypos) in enumerate(zip(x_coordinates, y_coordinates)):

            theta_x = [xpos]
            theta_y = [ypos]

            ray_x, ray_y, d = self._compute_comoving_ray_path(xpos, ypos, lens_model, kwargs_lens, zsource,
                               terminate_at_source, source_x, source_y)

            for rx, ry, di in zip(ray_x[1:], ray_y[1:], d[1:]):

                theta_x.append(rx/di)
                theta_y.append(ry/di)

            distances = [0.] + list(d[1:])
            distances = np.array(distances)
            theta_x = np.array(theta_x)
            theta_y = np.array(theta_y)

            angle_x.append(interp1d(distances, theta_x))
            angle_y.append(interp1d(distances, theta_y))

        if evaluate_at_mean:

            zrange = np.linspace(0., self.zsource, 100)
            comoving_distance_calc = self.cosmology.D_C_transverse
            distances = [comoving_distance_calc(zi) for zi in zrange]

            angular_coordinates_x = []
            angular_coordinates_y = []
            for di in distances:
                x_coords = [ray_x(di) for ray_x in angle_x]
                y_coords = [ray_y(di) for ray_y in angle_y]
                x_center = np.mean(x_coords)
                y_center = np.mean(y_coords)
                angular_coordinates_x.append(x_center)
                angular_coordinates_y.append(y_center)

            angle_x = [interp1d(distances, angular_coordinates_x)]
            angle_y = [interp1d(distances, angular_coordinates_y)]

        return angle_x, angle_y

    def _compute_comoving_ray_path(self, x_coordinate, y_coordinate, lens_model, kwargs_lens, zsource,
                              terminate_at_source=False, source_x=None, source_y=None):

        """
        :param x_coordinate: x coordinates to interpolate (arcsec) (float)
        :param y_coordinate: y coordinates to interpolate (arcsec) (float)
        Typically x_coordinates/y_coordinates would be four image positions, or the coordinate of the lens centroid
        :param lens_model: instance of LensModel (lenstronomy)
        :param kwargs_lens: keyword arguments for lens model
        :param zsource: source redshift
        :param terminate_at_source: fix the final angular coordinate to the source coordinate
        :param source_x: source x coordinate (arcsec)
        :param source_y: source y coordinate (arcsec)
        :return: Instance of interp1d (scipy) that returns the angular coordinate of a ray given a
        comoving distance
        """

        redshift_list = lens_model.redshift_list + [zsource]
        zstep = lenscone_default.default_z_step
        finely_sampled_redshifts = np.linspace(zstep, self.zsource - zstep, 50)
        all_redshifts = np.unique(np.append(redshift_list, finely_sampled_redshifts))

        all_redshifts_sorted = all_redshifts[np.argsort(all_redshifts)]

        comoving_distance_calc = self.cosmology.D_C_transverse

        x_start, y_start = 0., 0.
        z_start = 0.

        x_list = [0.]
        y_list = [0.]
        distances = [0.]

        alpha_x_start, alpha_y_start = x_coordinate, y_coordinate
        for zi in all_redshifts_sorted:

            x_start, y_start, alpha_x_start, alpha_y_start = lens_model.lens_model.ray_shooting_partial(x_start, y_start,
                                                                                alpha_x_start, alpha_y_start,
                                                                                z_start, zi, kwargs_lens)
            d = float(comoving_distance_calc(zi))
            x_list.append(x_start)
            y_list.append(y_start)
            distances.append(d)
            z_start = zi

        if terminate_at_source:
            d_src = comoving_distance_calc(self.zsource)
            x_list[-1] = source_x * d_src
            y_list[-1] = source_y * d_src

        return np.array(x_list), np.array(y_list), np.array(distances)

    def lens_plane_redshifts(self, kwargs_render=dict):

        zmin = lenscone_default.default_zstart
        if 'zstep' not in kwargs_render.keys():
            zstep = lenscone_default.default_z_step
        else:
            zstep = kwargs_render['zstep']

        front_z = np.arange(zmin, self.zlens, zstep)
        back_z = np.arange(self.zlens, self.zsource, zstep)
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

            if 'mass_function_model' not in self._halo_mass_function_args.keys():
                self._halo_mass_function_args.update({'mass_function_model': cosmo_default.default_mass_function})

            self.halo_mass_function = LensingMassFunction(self.cosmology, 10 ** logLOS_mlow, 10 ** logLOS_mhigh, self.zlens, self.zsource,
                                                          cone_opening_angle=args['cone_opening_angle'],
                                                          **self._halo_mass_function_args)

        return self.halo_mass_function

