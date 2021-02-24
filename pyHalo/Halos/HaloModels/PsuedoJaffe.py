from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.concentration import Concentration
import numpy as np
from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe

class PJaffeSubhalo(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        self._lens_cosmo = lens_cosmo_instance
        self._concentration = Concentration(lens_cosmo_instance)
        self._prof = PJaffe()
        super(PJaffeSubhalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    @property
    def params_physical(self):

        if not hasattr(self, '_params_physical'):

            [concentration] = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            _, rs_kpc, r200_kpc = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            ra_kpc = 0.01 * rs_kpc

            rho = self._rho(self.mass, rs_kpc, ra_kpc, r200)

            self._params_physical = {'rhos': rhos, 'rs': rs, 'r200': r200, 'rho': rho}

        return self._params_physical

    @property
    def lenstronomy_params(self):

        if not hasattr(self, '_lenstronomy_args'):

            kpc_to_arcsec = 1 / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            (concentration) = self.profile_args
            rhos_kpc, rs_kpc, r200_kpc = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            ra_kpc = 0.001 * rs_kpc

            rs_arcsec = rs_kpc * kpc_to_arcsec
            ra_arcsec = ra_kpc * kpc_to_arcsec
            r200_arcsec = r200_kpc * kpc_to_arcsec
            # fc = np.log(2) - 1/2
            # m_inside_rs = 4 * np.pi * rs_kpc ** 3 * rhos_kpc * fc
            #
            rho = self._rho(self.mass, rs_arcsec, ra_arcsec, r200_arcsec)

            sigma0 = self._prof.rho2sigma(rho, ra_arcsec, rs_arcsec)

            sigma_crit_Mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit = sigma_crit_Mpc * (0.001/kpc_to_arcsec) ** 2
            sigma_0 = sigma0/sigma_crit

            self._lenstronomy_args = {'center_x': self.x,
                                      'center_y': self.y,
                                      'Ra': ra_arcsec,
                                      'Rs': rs_arcsec,
                                      'sigma0': sigma_0}

        return self._lenstronomy_args, None

    def _rho(self, m, rs, ra, r_match):

        """
        returns the central density of a PJaffe halo such that the resulting halo has the mass m within r_match
        :param m:
        :param ra:
        :param r_match:
        :return:
        """
        f = (ra * np.arctan(r_match / ra) - rs * np.arctan(r_match / rs)) / (ra ** 2 - rs ** 2)
        rho = m / (4 * np.pi * ra ** 2 * rs ** 2 * f)
        return rho

    @property
    def lenstronomy_ID(self):
        return 'PJAFFE'

    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):

            if self._args['evaluate_mc_at_zlens']:
                z_eval = self.z
            else:
                z_eval = self.z_infall

            concentration = self._concentration.NFW_concentration(self.mass,
                                                                  z_eval,
                                                                  self._args['mc_model'],
                                                                  self._args['mc_mdef'],
                                                                  self._args['log_mc'],
                                                                  self._args['c_scatter'],
                                                                  self._args['c_scale'],
                                                                  self._args['c_power'],
                                                                  self._args['c_scatter_dex'])

            self._profile_args = (concentration)

        return self._profile_args

class PJaffeFieldhalo(PJaffeSubhalo):

    @property
    def profile_args(self):

        if not hasattr(self, '_profile_args'):

            concentration = self._concentration.NFW_concentration(self.mass,
                                                                  self.z,
                                                                  self._args['mc_model'],
                                                                  self._args['mc_mdef'],
                                                                  self._args['log_mc'],
                                                                  self._args['c_scatter'],
                                                                  self._args['c_scale'],
                                                                  self._args['c_power'],
                                                                  self._args['c_scatter_dex'])

            self._profile_args = (concentration)

        return self._profile_args
