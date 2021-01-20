import numpy as np
from pyHalo.defaults import lenscone_default
from scipy.interpolate import interp1d
from pyHalo.Cosmology.cosmology import Cosmology

def interpolate_ray_paths(x_coordinates, y_coordinates, lens_model, kwargs_lens, zsource,
                          terminate_at_source=False, source_x=None, source_y=None, evaluate_at_mean=False,
                          cosmo=None):

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

    if cosmo is None:
        cosmo = Cosmology()

    for i, (xpos, ypos) in enumerate(zip(x_coordinates, y_coordinates)):

        theta_x = [xpos]
        theta_y = [ypos]

        ray_x, ray_y, d = compute_comoving_ray_path(xpos, ypos, lens_model, kwargs_lens, zsource,
                                                          terminate_at_source, source_x, source_y, cosmo=cosmo)

        for rx, ry, di in zip(ray_x[1:], ray_y[1:], d[1:]):
            theta_x.append(rx / di)
            theta_y.append(ry / di)

        distances = [0.] + list(d[1:])
        distances = np.array(distances)
        theta_x = np.array(theta_x)
        theta_y = np.array(theta_y)

        angle_x.append(interp1d(distances, theta_x))
        angle_y.append(interp1d(distances, theta_y))

    if evaluate_at_mean:

        zrange = np.linspace(0., zsource, 100)
        comoving_distance_calc = cosmo.D_C_transverse
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

def compute_comoving_ray_path(x_coordinate, y_coordinate, lens_model, kwargs_lens, zsource,
                              terminate_at_source=False, source_x=None, source_y=None, cosmo=None):

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

        if cosmo is None:
            cosmo = Cosmology()

        redshift_list = lens_model.redshift_list + [zsource]
        zstep = lenscone_default.default_z_step
        finely_sampled_redshifts = np.linspace(zstep, zsource - zstep, 50)
        all_redshifts = np.unique(np.append(redshift_list, finely_sampled_redshifts))

        all_redshifts_sorted = all_redshifts[np.argsort(all_redshifts)]

        comoving_distance_calc = cosmo.D_C_transverse

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
            d_src = comoving_distance_calc(zsource)
            x_list[-1] = source_x * d_src
            y_list[-1] = source_y * d_src

        return np.array(x_list), np.array(y_list), np.array(distances)
