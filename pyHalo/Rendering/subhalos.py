import numpy as np
from copy import deepcopy
from pyHalo.Rendering.rendering_class_base import RenderingClassBase

class Subhalos(RenderingClassBase):

    """
    This class generates subhalos, objects that have been accreted onto the host halo of the main deflector.
    """

    def __init__(self, mass_function_model, kwargs_mass_function, spatial_distribution_model,
                 geometry, lens_cosmo):
        """

        :param mass_function_model:
        :param kwargs_mass_function:
        :param spatial_distribution_model:
        :param geometry:
        :param lens_cosmo:
        """
        kwargs_mass_function_internal = deepcopy(kwargs_mass_function)
        if 'log_m_host' in kwargs_mass_function.keys():
            kwargs_mass_function_internal['host_m200'] = 10 ** kwargs_mass_function['log_m_host']
            del kwargs_mass_function_internal['log_m_host']
        elif 'host_m200' in kwargs_mass_function.keys():
            pass
        else:
            raise Exception('must specify the host halo mass through keyword argument host_m200 or log_m_host (base 10)'
                            'when adding subhalos!')
        super(Subhalos, self).__init__(mass_function_model, kwargs_mass_function_internal, spatial_distribution_model,
                 geometry, lens_cosmo, None, None)

    def render(self):

        """
        Generates halo masses and positions for subhalos of the main deflector host halo.
        :return: mass (in Msun), x (arcsec), y (arcsec), r3d (kpc), redshift
        """
        mfunc_model = self._get_mass_function_model()
        m = mfunc_model.draw()
        x, y, r3d_kpc = self.render_positions_at_z(len(m))
        z = np.array([self._lens_cosmo.z_lens] * len(m))
        subhalo_flag = [True] * len(m)
        return m, x, y, r3d_kpc, z, subhalo_flag

    def render_positions_at_z(self, nhalos):

        """
        :param nhalos: number of halos or objects to generate
        :return: the x, y coordinate of objects in arcsec, and a 3 dimensional coordinate in kpc
        The 3d coordinate only has a clear physical interpretation for subhalos, and is used to compute truncation raddi.
        For line of sight halos it is set to None.
        """

        x_kpc, y_kpc, r3d_kpc = self._spatial_distribution_model.draw(nhalos, self._lens_cosmo.z_lens)
        if len(x_kpc) > 0:
            kpc_per_asec = self._geometry.kpc_per_arcsec(self._lens_cosmo.z_lens)
            x_arcsec = x_kpc * kpc_per_asec ** -1
            y_arcsec = y_kpc * kpc_per_asec ** -1
            return x_arcsec, y_arcsec, r3d_kpc
        else:
            return np.array([]), np.array([]), np.array([])

    def _get_mass_function_model(self, log_mlow=None, log_mhigh=None):
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
        kwargs_model['log_mlow'], kwargs_model['log_mhigh'] = \
            self._redshift_dependent_mass_range(self._lens_cosmo.z_lens, kwargs_model['log_mlow'],
                                                kwargs_model['log_mhigh'])
        kpc_per_asec_zlens = self._lens_cosmo.cosmo.kpc_proper_per_asec(self._lens_cosmo.z_lens)
        kwargs_model['power_law_index'] = self._kwargs_mass_function['power_law_index'] + \
                                          kwargs_model['delta_power_law_index']
        kwargs_model['normalization'] = normalization_sigmasub(kwargs_model['sigma_sub'],
                                                               kwargs_model['host_m200'], self._lens_cosmo.z_lens,
                                                               kpc_per_asec_zlens, self._geometry.cone_opening_angle,
                                                               kwargs_model['power_law_index'], kwargs_model['m_pivot'])
        return self._mass_function_model(**kwargs_model)

    def convergence_sheet_correction(self, kappa_scale, log_mlow, log_mhigh, *args, **kwargs):
        """

        :param kappa_scale:
        :param log_mlow:
        :param log_mhigh:
        :param args:
        :param kwargs:
        :return:
        """
        mass_in_subhalos = self._get_mass_function_model(log_mlow, log_mhigh).first_moment
        if mass_in_subhalos == 0:
            return [], [], []

        area = self._geometry.angle_to_physical_area(0.5 * self._geometry.cone_opening_angle, self._lens_cosmo.z_lens)
        kappa = mass_in_subhalos / self._lens_cosmo.sigma_crit_mass(self._lens_cosmo.z_lens, area)
        negative_kappa = -1 * kappa_scale * kappa
        kwargs_out = [{'kappa': negative_kappa}]
        profile_name_out = ['CONVERGENCE']
        redshifts_out = [self._lens_cosmo.z_lens]
        return kwargs_out, profile_name_out, redshifts_out

def host_scaling_function(mhalo, z, k1 = 0.88, k2 = 1.7):
    """

    :param mhalo:
    :param z:
    :param k1:
    :param k2:
    :return:
    """
    logscaling = k1 * np.log10(mhalo / 10**13) + k2 * np.log10(z + 0.5)

    return 10 ** logscaling


def normalization_sigmasub(sigma_sub, host_m200, zlens, kpc_per_asec_zlens, cone_opening_angle, plaw_index, m_pivot):
    """

    :param sigma_sub:
    :param host_m200:
    :param zlens:
    :param kpc_per_asec_zlens:
    :param cone_opening_angle:
    :param plaw_index:
    :param m_pivot:
    :return:
    """

    host_scaling = host_scaling_function(host_m200, zlens)
    a0_per_kpc2 = sigma_sub * host_scaling
    R_kpc = kpc_per_asec_zlens * (0.5 * cone_opening_angle)
    area = np.pi * R_kpc ** 2
    m_pivot_factor = m_pivot ** -(plaw_index + 1)
    normalization = area * a0_per_kpc2 * m_pivot_factor
    return normalization
