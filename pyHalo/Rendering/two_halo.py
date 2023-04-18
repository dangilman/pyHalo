import numpy as np
from copy import deepcopy
from pyHalo.Rendering.line_of_sight import LineOfSightNoSheet

class TwoHaloContribution(LineOfSightNoSheet):

    """
    This class adds correlated structure associated with the host dark matter halo. The amount of structure added is
    proportional to b * corr, where b is the halo bias as computed by Sheth and Tormen (1999) and corr is the
    matter-matter correlation function. Currently, this term is implemented as a rescaling of the background density by
    b * corr, where the product is the average value computed over 2*dz, where dz is the spacing of the redshift planes
    adjacent the redshift plane of the main deflector.
    """

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
        if 'host_m200' in kwargs_mass_function.keys():
            host_m200 = kwargs_mass_function['host_m200']
        elif 'log_m_host' in kwargs_mass_function.keys():
            host_m200 = 10 ** kwargs_mass_function['log_m_host']
        else:
            raise Exception('must specify the host halo mass through keyword argument host_m200 or log_m_host (base 10)'
                            'when adding the two-halo term!')
        z_eval = lens_cosmo.z_lens
        idx = np.argmin(abs(np.array(lens_plane_redshifts) - z_eval))
        delta_z = delta_z_list[idx]
        rmax = lens_cosmo.cosmo.D_C_transverse(z_eval + delta_z) - lens_cosmo.cosmo.D_C_transverse(z_eval)
        rmin = min(rmax, 0.5)
        two_halo_boost = self._lens_cosmo.two_halo_boost(host_m200, z_eval, rmin=rmin, rmax=rmax)
        line_of_sight_rescaling = two_halo_boost - 1.
        kwargs_mass_function_scaled = deepcopy(kwargs_mass_function)
        kwargs_mass_function_scaled['LOS_normalization'] = line_of_sight_rescaling
        super(TwoHaloContribution, self).__init__(mass_function_model, kwargs_mass_function_scaled, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshifts, delta_z_list)

