import numpy as np
from copy import deepcopy
from pyHalo.Rendering.rendering_class_base import RenderingClassBase


class LineOfSightNoSheet(RenderingClassBase):
    """
    This class generates halos between the observer and source that are
    not bound to the host dark matter halo around the main deflector.
    """

    def render(self):

        """
        Generates halo masses and positions for objects along the line of sight
        (except for halos from the two-halo contribution)
        :return: mass (in Msun), x (arcsec), y (arcsec), r3d (kpc), redshift
        """

        masses = np.array([])
        x = np.array([])
        y = np.array([])
        redshifts = np.array([])

        for z, dz in zip(self._lens_plane_redshifts, self._delta_z_list):
            mfunc_model = self._get_mass_function_model(z, dz)
            m = mfunc_model.draw()
            nhalos = len(m)
            _x, _y = self.render_positions_at_z(z, nhalos)
            _z = np.array([z] * len(_x))
            masses = np.append(masses, m)
            x = np.append(x, _x)
            y = np.append(y, _y)
            redshifts = np.append(redshifts, _z)

        subhalo_flag = [False] * len(masses)
        r3d = np.array([None] * len(masses))
        return masses, x, y, r3d, redshifts, subhalo_flag

    def _get_mass_function_model(self, z, delta_z, log_mlow=None, log_mhigh=None):
        """

        :param z:
        :param delta_z:
        :param log_mlow: replaces log_mlow in kwargs_mass_function if not None
        :param log_mhigh: replaces log_mhigh in kwargs_mass_function if not None
        :return:
        """
        kwargs_model = deepcopy(self._kwargs_mass_function)
        if log_mlow is not None:
            kwargs_model['log_mlow'] = log_mlow
        if log_mhigh is not None:
            kwargs_model['log_mhigh'] = log_mhigh
        kwargs_model['log_mlow'], kwargs_model['log_mhigh'] = self._redshift_dependent_mass_range(z,
                                                                       kwargs_model['log_mlow'],
                                                                       kwargs_model['log_mhigh'])
        mfunc_model = self._mass_function_model.from_redshift(z, delta_z, self._geometry, kwargs_model)
        return mfunc_model


class LineOfSight(LineOfSightNoSheet):

    """
    This class generates halos between the observer and source that are not bound to the host dark matter halo around
    the main deflector, with the inclusion of negative sheets of convergence to remove the mass added in halos
    """

    def convergence_sheet_correction(self, kappa_scale, log_mlow, log_mhigh, zmin=None, zmax=None, *args, **kwargs):

        """
        this routine applies the negative convergence sheet correction for lens planes along the line of sight
        :param kwargs_mass_sheets: keyword arguments that overwrite whatever the default settings for the mass sheet
        sheet are - leave it as None for most applications
        :return:
        """

        lens_plane_redshifts = self._lens_plane_redshifts
        delta_zs = self._delta_z_list
        kwargs_out = []
        profile_names_out = []
        redshifts = []

        if zmin is None:
            zmin = 0.0
        if zmax is None:
            zmax = self._geometry._zsource

        for z, delta_z in zip(lens_plane_redshifts, delta_zs):

            if z < zmin:
                continue
            if z > zmax:
                continue

            log_mass_sheet_correction_min, log_mass_sheet_correction_max = self._redshift_dependent_mass_range(
                z, log_mlow, log_mhigh)
            mass_function_model = self._get_mass_function_model(z, delta_z,
                                                                log_mlow=log_mass_sheet_correction_min,
                                                                log_mhigh=log_mass_sheet_correction_max)
            first_moment = mass_function_model.first_moment
            kappa = self._convergence_at_z(z, self._geometry, self._lens_cosmo, first_moment)
            if kappa > 0:
                kwargs_out.append({'kappa': - kappa_scale * kappa})
                profile_names_out += ['CONVERGENCE']
                redshifts.append(z)
        return kwargs_out, profile_names_out, redshifts

    @staticmethod
    def _convergence_at_z(z, geometry_class, lens_cosmo_class, mtheory):

        area = geometry_class.angle_to_physical_area(0.5 * geometry_class.cone_opening_angle, z)
        sigma_crit_mass = lens_cosmo_class.sigma_crit_mass(z, area)
        return mtheory / sigma_crit_mass
