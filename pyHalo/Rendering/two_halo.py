import numpy as np
from copy import deepcopy
from pyHalo.Rendering.line_of_sight import LineOfSightNoSheet
from scipy.integrate import quad
from pyHalo.concentration_models import ConcentrationDiemerJoyce


class TwoHaloContribution(LineOfSightNoSheet):

    name = 'TWO_HALO_TERM'
    """
    This class adds correlated structure associated with the host dark matter halo. The amount of structure added is
    proportional to b * corr, where b is the halo bias as computed by Sheth and Tormen (1999) and corr is the
    matter-matter correlation function. Currently, this term is implemented as a rescaling of the background density by
    b * corr, where the product is the average value computed over 2*dz, where dz is the spacing of the redshift planes
    adjacent the redshift plane of the main deflector.
    """

    def __init__(self, mass_function_model, kwargs_mass_function, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshifts, delta_z_list, use_Lazar_correction=True,
                 scale_2halo_boost_factor=1.0):

        """
        Adds additional halos from correlated structure around the main deflector

        :param mass_function_model: a mass fucntion class to generate halos (see Rendering/MassFunctions)
        :param kwargs_mass_function: keyword arguments to pass to the mass function
        :param spatial_distribution_model: keyword arguments to pass to the spatial distribution class
        :param geometry: an instance of Geometry class
        :param lens_cosmo: an instance of LensCosmo class
        :param lens_plane_redshifts: a list of redshifts corresponding to lens planes along the LOS
        :param use_Lazar_correction: bool; apply the correction by Lazar et al. to increase the number of halos
        :param delta_z_list: spacing between LOS lens planes
        :param use_Lazar_correction: bool; apply the correction by Lazar et al.
        :param scale_2halo_boost_factor: float; scale factor for 2halo boost
        """
        if 'host_m200' in kwargs_mass_function.keys():
            host_m200 = kwargs_mass_function['host_m200']
        elif 'log_m_host' in kwargs_mass_function.keys():
            host_m200 = 10 ** kwargs_mass_function['log_m_host']
        else:
            raise Exception('must specify the host halo mass through keyword argument host_m200 or log_m_host (base 10)'
                            'when adding the two-halo term!')
        concentration_model = ConcentrationDiemerJoyce(lens_cosmo.cosmo.astropy, scatter=False)
        c_host = concentration_model.nfw_concentration(host_m200, lens_cosmo.z_lens)
        _, _, r200_host_kpc = lens_cosmo.NFW_params_physical(host_m200, c_host, lens_cosmo.z_lens)
        r200_host_mpc = r200_host_kpc * 1e-3
        z_eval = lens_cosmo.z_lens
        idx = np.argmin(abs(np.array(lens_plane_redshifts) - z_eval))
        delta_z = delta_z_list[idx]
        boost = two_halo_enhancement_factor(z_eval, delta_z, lens_cosmo, host_m200, r200_host_mpc, use_Lazar_correction,
                                            scale_2halo_boost_factor)
        kwargs_mass_function_scaled = deepcopy(kwargs_mass_function)
        kwargs_mass_function_scaled['LOS_normalization'] *= boost
        self._delta_z = delta_z
        super(TwoHaloContribution, self).__init__(mass_function_model, kwargs_mass_function_scaled, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshifts, delta_z_list)

    def render(self):
        """
        Generates halo masses and positions for correlated structure near the main deflector.
        These objects are placed at the lens redshift
        :return: mass (in Msun), x (arcsec), y (arcsec), r3d (kpc), redshift, subhalo_flag (bool)
        """
        mfunc_model = self._get_mass_function_model(self._lens_cosmo.z_lens, self._delta_z)
        masses = mfunc_model.draw()
        nhalos = len(masses)
        x, y = self.render_positions_at_z(self._lens_cosmo.z_lens, nhalos)
        redshifts = np.array([self._lens_cosmo.z_lens] * nhalos)
        subhalo_flag = [False] * nhalos
        r3d = np.array([None] * nhalos)
        return masses, x, y, r3d, redshifts, subhalo_flag

def modificationLazar2021(r, r200, b_e=0.1, s_e=4.0):
    """
    Modification to the two-halo contribution around the main deflector using Equation 3 from Lazar et al. (2021)
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.6064L/abstract
    :param r: distance in Mpc
    :param r200: host halo virial radius in Mpc
    :param b_e: calibrated parameter
    :param s_e: calibrated parameter
    :return: the correction factor for the background density that gives a better match to simulations
    """
    return b_e * (r / r200 / 5) ** -s_e

def _boost_integrand(r, m200_host, z, r200_host, lens_cosmo, use_Lazar_correction):
    """
    The local enhancement of the background density of the Universe from the two-halo term and a correction factor
    see Lazar et al. 2021
    :param r: distance from the host in Mpc
    :param m200_host: host halo mass
    :param z: host halo redshift
    :param r200_host: host halo virial radius
    :param lens_cosmo: an instance of LensCosmo
    :return: the local enhancement to the background density
    """
    if use_Lazar_correction:
        return lens_cosmo.twohaloterm(r, m200_host, z) + modificationLazar2021(r, r200_host)
    else:
        return lens_cosmo.twohaloterm(r, m200_host, z)

def two_halo_enhancement_factor(z_lens, z_step, lens_cosmo, overdensity_m200, r200_host, use_Lazar_correction,
                                scale_2halo_boost_factor=1.0):
    """
    Calculates the enhancement of the background density due to correlated structure around the main deflector. Includes
    a contribution from the two-halo term and a correction factor calibrated by Lazar et al. (2021)
    :param z_lens: deflector redshift
    :param z_step: redshift spacing
    :param lens_cosmo: instance of LensCosmo
    :param overdensity_m200: host halo mass
    :param use_Lazar_correction: bool; adds the correction derived by Lazar et al. (2021)
    :param scale_2halo_boost_factor: factor by which to rescale the two-halo enhancement factor
    :return: the enhancement of the background density per unit length
    """
    rmax = lens_cosmo.cosmo.D_C_transverse(z_lens + z_step) - lens_cosmo.cosmo.D_C_transverse(z_lens)
    rmin = min(rmax, 0.5)
    args = (overdensity_m200, z_lens, r200_host, lens_cosmo, use_Lazar_correction)
    two_halo_boost = 2 * quad(_boost_integrand, rmin, rmax, args=args)[0] / (rmax - rmin)
    return scale_2halo_boost_factor * two_halo_boost
