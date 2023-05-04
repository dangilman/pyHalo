import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from pyHalo.Rendering.rendering_class_base import RenderingClassBase
from pyHalo.Rendering.SpatialDistributions.correlated import Correlated2D
from pyHalo.Rendering.MassFunctions.delta_function import DeltaFunction
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.single_realization import realization_at_z

class CorrelatedStructure(RenderingClassBase):

    """
    This class generates a population of halos with a spatial distribution that tracks the dark matter density in halos
    at each lens plane
    """
    def __init__(self, mass_function_model, kwargs_mass_function,
                 geometry, lens_cosmo, lens_plane_redshifts, delta_z_list, realization):

        self._cylinder_geometry = Geometry(lens_cosmo.cosmo, lens_cosmo.z_lens, lens_cosmo.z_source,
                                           1.0, 'CYLINDER')
        spatial_distribution_model = Correlated2D(self._cylinder_geometry)
        super(CorrelatedStructure, self).__init__(mass_function_model, kwargs_mass_function, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshifts, delta_z_list)
        self._realization = realization

    def render(self, rmax, x_center_interp, y_center_interp, arcsec_per_pixel):

        """
        Generates halo masses and positions for correlated structure along the line of sight around
        the angular coordinate of each light ray
        :r_max: the maximum radius in arcsec around (x_center, y_center) around which to generate halos
        :param x_center_interp: an interp1d function that returns the x angular position of a
        ray given a comoving distance
        :param y_center_interp: an interp1d function that returns the y angular position of a
        ray given a comoving distance
        :param arcsec_per_pixel: sets the spatial resolution for the rendering of correlated structure
        :return: mass (in Msun), x (arcsec), y (arcsec), r3d (kpc), redshift
        """

        masses = np.array([])
        x = np.array([])
        y = np.array([])
        redshifts = np.array([])

        plane_redshifts = self._realization.unique_redshifts
        delta_z = []
        rescale_inds = []
        rescale_factors = []

        for i, zi in enumerate(plane_redshifts[0:-1]):
            delta_z.append(plane_redshifts[i + 1] - plane_redshifts[i])
        delta_z.append(self._realization.lens_cosmo.z_source - plane_redshifts[-1])

        for z, dz in zip(plane_redshifts, delta_z):

            #rendering_radius = rmax * self._cylinder_geometry.rendering_scale(z)
            d = self._lens_cosmo.cosmo.D_C_transverse(z)
            x_angle = x_center_interp(d)
            y_angle = y_center_interp(d)
            _m, _x, _y, halo_inds, rescale_factor = self.render_at_z(z, x_angle, y_angle,
                                                rmax, arcsec_per_pixel)
            if len(_m) > 0:
                _z = np.array([z] * len(_x))
                masses = np.append(masses, _m)
                x = np.append(x, _x)
                y = np.append(y, _y)
                redshifts = np.append(redshifts, _z)
                rescale_inds += halo_inds
                rescale_factors += [rescale_factor] * len(halo_inds)

        subhalo_flag = [False] * len(masses)
        r3d = np.array([None] * len(masses))

        return masses, x, y, r3d, redshifts, subhalo_flag, rescale_inds, rescale_factors

    def render_at_z(self, z, angular_coordinate_x, angular_coordinate_y, rendering_radius, arcsec_per_pixel):

        """

        :param n: number of objects to render
        :param z: redshift
        :param angular_coordinate_x: the angular coordinate in arcsec of a light ray at redshift z
        :param angular_coordinate_y: the angular coordinate in arcsec of a light ray at redshift z
        :param rendering_radius: the angular radius inside which to render objects
        :param arcsec_per_pixel: sets the spatial resolution for the rendering of correlated structure
        :return: the positions in arcsec of the rendered objects
        """

        kpc_per_asec = self._cylinder_geometry.kpc_per_arcsec(z)

        pdf, mass_in_area, halo_indexes = self._kappa_at_lens_plane(z, angular_coordinate_x, angular_coordinate_y, rendering_radius,
                                                      arcsec_per_pixel)

        if np.sum(pdf) == 0:
            return np.array([]), np.array([]), np.array([]), [], 1.

        m, rescale_factor = self.render_masses_at_z(mass_in_area)

        n_halos = len(m)
        if n_halos > 0:
            x_kpc, y_kpc = self._spatial_distribution_model.draw(n_halos, rendering_radius, pdf, z,
                                                                angular_coordinate_x, angular_coordinate_y)


            x_arcsec = x_kpc / kpc_per_asec
            y_arcsec = y_kpc / kpc_per_asec
            return m, x_arcsec, y_arcsec, halo_indexes, rescale_factor
        else:
            return np.array([]), np.array([]), np.array([]), [], 1.

    def render_masses_at_z(self, mass_in_area):

        """
        :param mass_in_area: total mass at the lens plane
        :return: halo masses at the desired redshift in units Msun, and the factor by which to rescale the
        original halo profiles
        """

        if self._mass_function_model.name == 'DELTA_FUNCTION':
            rescale_factor = 1.-self._kwargs_mass_function['mass_fraction']
            volume = 1.
            rho = self._kwargs_mass_function['mass_fraction'] * mass_in_area
            mass = 10 ** self._kwargs_mass_function['logM']
            mass_function = self._mass_function_model(mass, volume, rho, self._kwargs_mass_function['draw_poisson'])
        else:
            raise Exception('this class is only implemented for a delta function mass function')

        return mass_function.draw(), rescale_factor

    def _kappa_at_lens_plane(self, z, angular_coordinate_x, angular_coordinate_y,
                            rendering_radius, arcsec_per_pixel):

        """
        This routine computes the 2D convergence map of all halos at a certain redshift centered around (angular_coordinate_x,
        angular_coordinate_y) out to a radius "rendering_radius"

        :param z: redshift
        :param angular_coordinate_x: x center in arcsec
        :param angular_coordinate_y: y center in arcsec
        :param rendering_radius: radius in arcsec
        :param arcsec_per_pixel: pixel size in arcsec
        :return: the convergence map, the total mass in the convergence map in the specified area,
        the indexes of halos contributing to the projected mass map
        """
        realization_at_plane, halo_indexes = realization_at_z(self._realization,
                                                   z,
                                                   angular_coordinate_x,
                                                   angular_coordinate_y,
                                                   2*rendering_radius,
                                                   mass_sheet_correction=False)

        lens_model_list, _, kwargs_lens, numerical_interp = realization_at_plane.lensing_quantities(
            add_mass_sheet_correction=False)

        if len(lens_model_list) == 0:
            return np.array([]), np.array([]), []

        lens_model = LensModel(lens_model_list, numerical_alpha_class=numerical_interp)
        npix = int(2 * rendering_radius / arcsec_per_pixel)
        _r = np.linspace(-rendering_radius, rendering_radius, npix)
        xx, yy = np.meshgrid(_r, _r)
        shape0 = xx.shape
        xx, yy = xx.ravel(), yy.ravel()
        rr = np.sqrt(xx ** 2 + yy ** 2)
        inds_zero = np.where(rr > rendering_radius)[0].ravel()
        pdf = lens_model.kappa(xx + angular_coordinate_x, yy + angular_coordinate_y, kwargs_lens)
        pdf[inds_zero] = 0.
        inds_nan = np.where(np.isnan(pdf))
        pdf[inds_nan] = 0.
        npixels = len(inds_zero)
        rendering_radius_mpc = rendering_radius * (0.001 * self._cylinder_geometry.kpc_per_arcsec(z))
        effective_area = np.pi * rendering_radius_mpc ** 2 / npixels
        mass_in_area = self._mass_in_area(pdf, z, effective_area)
        return pdf.reshape(shape0), mass_in_area, halo_indexes

    def _mass_in_area(self, kappa_pdf, z, area):

        """

        :param kappa_pdf: a convergence map at the given redshift
        :param z: redshift
        :param area: the "area" of the convergence map in Mpc^2/pixel
        :return: the total mass contained in the convergence map
        """
        sigma_crit = self._realization.lens_cosmo.get_sigma_crit_lensing(
            z, self._realization.lens_cosmo.z_source)
        mass_in_area = np.sum(kappa_pdf * sigma_crit) * area
        return mass_in_area

