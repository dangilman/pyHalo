import numpy as np


class RenderingClassBase(object):

    def __init__(self, mass_function_model, kwargs_mass_function, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshifts, delta_z_list):

        """

        :param mass_function_model:
        :param kwargs_mass_function:
        :param spatial_distribution_model:
        :param geometry:
        :param lens_cosmo:
        :param lens_plane_redshifts:
        :param delta_z_list:
        """
        self._mass_function_model = mass_function_model
        self._spatial_distribution_model = spatial_distribution_model
        if 'draw_poisson' not in kwargs_mass_function.keys():
            kwargs_mass_function['draw_poisson'] = True
        self._kwargs_mass_function = kwargs_mass_function
        self._geometry = geometry
        self._lens_cosmo = lens_cosmo
        self._lens_plane_redshifts = lens_plane_redshifts
        self._delta_z_list = delta_z_list

    def render_positions_at_z(self, z, nhalos):

        """
        :param z: redshift
        :param nhalos: number of halos or objects to generate
        :return: the x, y coordinate of objects in arcsec, and a 3 dimensional coordinate in kpc
        The 3d coordinate only has a clear physical interpretation for subhalos, and is used to compute truncation raddi.
        For line of sight halos it is set to None.
        """

        x_kpc, y_kpc = self._spatial_distribution_model.draw(nhalos, z)
        if len(x_kpc) > 0:
            kpc_per_asec = self._geometry.kpc_per_arcsec(z)
            x_arcsec = x_kpc * kpc_per_asec ** -1
            y_arcsec = y_kpc * kpc_per_asec ** -1
            return x_arcsec, y_arcsec
        else:
            return np.array([]), np.array([])

    @staticmethod
    def _redshift_dependent_normalization(z, normalization):
        """
        Evaluates a possibly redshift-dependent line-of-sight mass function normalization
        :param z: redshift
        :param normalization: the normalization passed to create the realization
        :return:
        """
        if callable(normalization):
            norm = normalization(z)
        else:
            norm = normalization
        return norm

    @staticmethod
    def _redshift_dependent_mass_range(z, log_mlow_object, log_mhigh_object):
        """
        Evaluates a possibly redshift-dependent minimum/maximum halo mass
        :param z: redshift
        :param log_mlow_object: either a number representing the minimum halo mass, or a callable function that returns
        log10(M_min) as a function z
        :param log_mhigh_object: either a number representing the maximum halo mass, or a callable function that returns
        log10(M_max) as a function z
        :return: the minimum and maximum halo mass (in log10)
        """

        if callable(log_mlow_object):
            log_mlow = log_mlow_object(z)
        else:
            log_mlow = log_mlow_object

        if callable(log_mhigh_object):
            log_mhigh = log_mhigh_object(z)
        else:
            log_mhigh = log_mhigh_object

        return log_mlow, log_mhigh

    def convergence_sheet_correction(self, *args, **kwargs):
        return [], [], []
