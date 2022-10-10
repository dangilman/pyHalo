from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform
import numpy as np
from copy import deepcopy
from pyHalo.Rendering.MassFunctions.power_law import GeneralPowerLaw
from pyHalo.Rendering.rendering_class_base import Rendering

class TwoHaloContribution(Rendering):

    """
    This class adds correlated structure associated with the host dark matter halo. The amount of structure added is
    proportional to b * corr, where b is the halo bias as computed by Sheth and Tormen (1999) and corr is the
    matter-matter correlation function. Currently, this term is implemented as a rescaling of the background density by
    b * corr, where the product is the average value computed over 2*dz, where dz is the spacing of the redshift planes
    adjacent the redshift plane of the main deflector.
    """
    def __init__(self, keywords_master, halo_mass_function, geometry, lens_cosmo, lens_plane_redshifts, delta_z_list):

        self._rendering_kwargs = self.keyword_parse_render(keywords_master)
        self.halo_mass_function = halo_mass_function
        self.geometry = geometry
        self.lens_cosmo = lens_cosmo
        self.spatial_distribution_model = LensConeUniform(keywords_master['cone_opening_angle'], geometry)
        self._lens_plane_redshifts = lens_plane_redshifts
        self._delta_z_list = delta_z_list
        super(TwoHaloContribution, self).__init__(keywords_master)

    def render(self):

        """
        Generates halo masses and positions for correlated structure around the main deflector
        :return: mass (in Msun), x (arcsec), y (arcsec), r3d (kpc), redshift
        """

        idx = np.argmin(abs(np.array(self._lens_plane_redshifts) - self.lens_cosmo.z_lens))
        delta_z = self._delta_z_list[idx]
        m = self.render_masses_at_z(self.lens_cosmo.z_lens, delta_z)
        x, y = self.render_positions_at_z(self.lens_cosmo.z_lens, len(m))
        subhalo_flag = [False] * len(m)
        redshifts = [self.lens_cosmo.z_lens] * len(m)
        r3d = np.array([None] * len(m))

        return m, x, y, r3d, redshifts, subhalo_flag

    def render_masses_at_z(self, z, delta_z):

        """
        :param z: redshift at which to render masses
        :param delta_z: thickness of the redshift slice
        :return: halo masses at the desired redshift in units Msun
        """

        norm, slope = self._norm_slope(z, delta_z)
        args = deepcopy(self._rendering_kwargs)
        log_mlow, log_mhigh = self._redshift_dependent_mass_range(z, args['log_mlow'], args['log_mhigh'])
        mfunc = GeneralPowerLaw(log_mlow, log_mhigh, slope, args['draw_poisson'],
                                norm, self._mass_function_model_util, self._kwargs_mass_function_model)
        m = mfunc.draw()

        return m

    def render_positions_at_z(self, z, nhalos):

        """
        :param z: redshift
        :param nhalos: number of halos or objects to generate
        :return: the x, y coordinate of objects in arcsec, and a 3 dimensional coordinate in kpc
        The 3d coordinate only has a clear physical interpretation for subhalos, and is used to compute truncation raddi.
        For line of sight halos it is set to None.
        """

        x_kpc, y_kpc = self.spatial_distribution_model.draw(nhalos, z)
        if len(x_kpc) > 0:
            kpc_per_asec = self.geometry.kpc_per_arcsec(z)
            x_arcsec = x_kpc * kpc_per_asec ** -1
            y_arcsec = y_kpc * kpc_per_asec ** -1
            return x_arcsec, y_arcsec

        else:
            return np.array([]), np.array([])

    def _norm_slope(self, z, delta_z):

        """
        This method computes the normalization of the mass function for correlated structure around the main deflector.

        The normalization is defined as (boost - 1) * background, where background is the mean normalization of the
        halo mass function computed with (for example) Sheth-Tormen, and boost is the average contribution of the
        two-halo term integrated over a comoving distance corresponding to 2 * dz, where dz is the redshift plane
        spacing.

        boost(z, r_min, r_max) = 2 / r_max int_{r_min}^{r_max} x(r, z, M_{host}) * dr

        where xi(r, M_{host) is the linear halo bias times the matter-matter correlation function,
        r_min is set of 0.5 Mpc, and r_max is the comoving distance corresponding to 2*dz, where dz is the redshift
        spacing. M_host is the mass in M_sun of the host dark matter halo
        :param z: the redshift which to evaluate the matter-matter correlation function and halo bias
        :param delta_z: the redshift spacing of the lens planes adjacent the main deflector
        :return: the normalization of the two-halo term mass function. The form of the two-halo term mass function is
        assumed to have the same shape as the background halo mass function
        """

        if z != self.lens_cosmo.z_lens:
            raise Exception('this class must be evaluated at the main deflector redshift')

        los_norm = self._redshift_dependent_normalization(z, self._rendering_kwargs['LOS_normalization'])
        volume_element_comoving = self.geometry.volume_element_comoving(z, delta_z)
        plaw_index = self.halo_mass_function.plaw_index_z(z) + self._rendering_kwargs['delta_power_law_index']
        norm_per_unit_volume = self.halo_mass_function.norm_at_z_density(z, plaw_index,
                                                                         self._rendering_kwargs['m_pivot'])
        norm_per_unit_volume *= los_norm
        reference_norm = norm_per_unit_volume * volume_element_comoving

        rmax = self.lens_cosmo.cosmo.D_C_transverse(z + delta_z) - self.lens_cosmo.cosmo.D_C_transverse(z)
        rmin = min(rmax, 0.5)

        two_halo_boost = self.halo_mass_function.two_halo_boost(self._rendering_kwargs['host_m200'], z, rmax=rmax,
                                                                rmin=rmin)

        slope = self.halo_mass_function.plaw_index_z(z) + self._rendering_kwargs['delta_power_law_index']
        norm = (two_halo_boost - 1) * reference_norm
        return norm, slope

    def convergence_sheet_correction(self, *args, **kwargs):

        return {}, [], []

    @staticmethod
    def keyword_parse_render(keywords_master):

        kwargs = {}
        required_keys = ['log_mlow', 'log_mhigh', 'host_m200', 'LOS_normalization',
                         'draw_poisson', 'delta_power_law_index', 'm_pivot', 'log_mc', 'a_wdm', 'b_wdm', 'c_wdm']

        for key in required_keys:

            if key not in keywords_master:
                raise Exception('Required keyword argument ' + str(key) + ' not specified.')
            else:
                kwargs[key] = keywords_master[key]

        return kwargs

    def keys_convergence_sheets(self):

        return {}
