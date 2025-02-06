from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import PseudoDoublePowerlaw
import numpy as np

class PrompCusp(Halo):
    """
    The base class for a prompt cusp
    """
    _gamma_inner = 1.5
    _gamma_outer = 10.0
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._prof = PseudoDoublePowerlaw()
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        mdef = 'PSEUDO_DPL'
        super(PrompCusp, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                              lens_cosmo_instance, args, unique_tag)

    def density_profile_3d(self, r):
        """

        :param r:
        :return:
        """
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        rho0_arcsec = self.density_normalization
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
        rho0_kpc = rho0_arcsec * sigma_crit_arcsec / kpc_per_arcsec ** 3
        rs_kpc = self._args['cusp_R'] * 1000
        return self._prof.density(r, rs_kpc, rho0_kpc, self._gamma_inner, self._gamma_outer)

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):

            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            x, y = np.round(self.x, 4), np.round(self.y, 4)
            rho0 = self.density_normalization
            R = self._args['cusp_R'] * 0.001
            R = np.round(R / kpc_per_arcsec, 10)
            alpha_Rs = self._prof.rho02alpha(rho0, R, self._gamma_inner, self._gamma_outer)
            alpha_Rs = np.round(alpha_Rs, 10)
            self._lenstronomy_args = [{'alpha_Rs': self._rescale_norm * alpha_Rs,
                                       'Rs': R,
                                       'gamma_inner': self._gamma_inner,
                                       'center_x': x,
                                       'center_y': y,
                                      'gamma_outer': self._gamma_outer}]
        return self._lenstronomy_args, None

    @property
    def profile_args(self):
        """

        :return:
        """
        return (self.density_normalization, self._args['cusp_R'], self._args['cusp_A'])

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['PSEUDO_DPL']

    @property
    def density_normalization(self):
        """
        Calculates the density normalization that conserves mass within x_match
        :return:
        """
        if not hasattr(self, '_rho0_norm'):

            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
            R_mpc = self._args['cusp_R'] # in mpc
            cuspA = self._args['cusp_A'] # M_sun * mpc^-1.5
            R_kpc = R_mpc * 1000
            mass = 8 * np.pi / 3 * cuspA * R_mpc ** 1.5
            self._rho0_norm = mass / self._prof.mass_3d(R_kpc/kpc_per_arcsec,
                                                        R_kpc/kpc_per_arcsec,
                                                     sigma_crit_arcsec,
                                                     self._gamma_inner,
                                                     self._gamma_outer)
        return self._rho0_norm
