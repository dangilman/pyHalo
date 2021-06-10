from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.concentration import Concentration
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.uldm import Uldm 
import lenstronomy.Util.constants as const
import numpy as np

class ULDMFieldHalo(Halo):

    """
    The base class for a truncated NFW halo
    """
    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        self._concentration = Concentration(lens_cosmo_instance)

        super(ULDMFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                           lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        return ['TNFW', 'ULDM']

    @property
    def c(self):
        """
        Computes the NFW halo concentration (once)
        """
        if not hasattr(self, '_c'):
            self._c = self._concentration.NFW_concentration(self.mass,
                                                                  self.z_eval,
                                                                  self._args['mc_model'],
                                                                  self._args['mc_mdef'],
                                                                  self._args['log_mc'],
                                                                  self._args['c_scatter'],
                                                                  self._args['c_scale'],
                                                                  self._args['c_power'],
                                                                  self._args['c_scatter_dex'])
        return self._c

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        # Copied from the TNFW class
        [concentration, rt, theta_c, kappa_0] = self.profile_args
        Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)

        x, y = np.round(self.x, 4), np.round(self.y, 4)

        Rs_angle = np.round(Rs_angle, 10)
        theta_Rs = np.round(theta_Rs, 10)
        r_trunc_arcsec = rt / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

        kwargs_nfw_temporary = {'alpha_Rs': theta_Rs, 'Rs': Rs_angle,
                  'center_x': x, 'center_y': y, 'r_trunc': r_trunc_arcsec}

        # need to specify the keyword arguments for the ULDM profile and renormalize NFW profile
        
        kwargs_uldm = {'theta_c': theta_c, 'kappa_0': kappa_0,
                  'center_x': x, 'center_y': y}

        kwargs_nfw = self._rescaled_nfw_params(self._args['R_max'], kwargs_nfw_temporary, kwargs_uldm)

        if kwargs_nfw['alpha_Rs'] < 0:
            raise ValueError('The resulting composite NFW profile mass is negative, tweak your parameters.')
        else:
            pass

        kwargs = [kwargs_nfw, kwargs_uldm]

        return kwargs, None
    
    @property
    def z_eval(self):
        """
        Returns the halo redshift
        """
        return self.z

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):

            truncation_radius = self._lens_cosmo.LOS_truncation_rN(self.mass, self.z,
                                                             self._args['LOS_truncation_factor'])

            [theta_c, kappa_0] = self._uldm_args(self._args['m_uldm'], self._args['M_uldm'], self._args['uldm_power_law'])

            self._profile_args = (self.c, truncation_radius, theta_c, kappa_0)

        return self._profile_args
    
    def _uldm_args(self, m, M, plaw):
        """
        Returns core radius in arcsec
        """
        m_log10 = np.log10(m)
        M_log10 = np.log10(M)

        a = 1/(1+self.z)
        m22 = 10**(m_log10 + 22)
        M9 = 10**(M_log10 - 9)

        Sigma_crit = self._lens_cosmo.sigma_crit_lensing * 1e-12
        D_lens = self._lens_cosmo.D_d * 1e6

        r_c = 160 * m22**(-1) * a**(0.5) * m22**(-1) * M9**(-plaw) 
        rho_c = 190 * a**(-1) * m22**(-2) * (r_c/100)**(-4)

        theta_c = r_c / D_lens / const.arcsec 
        kappa_0 = 429 * np.pi * rho_c * r_c / (2048 * np.sqrt(0.091) * Sigma_crit)

        return [theta_c, kappa_0]
    
    def _rescaled_nfw_params(self, R_max, nfw_params, uldm_params):
        """
        Returns rescaled NFW params
        """
        R_max_Rs = R_max * nfw_params['Rs']
        M_nfw = NFW().mass_3d_lens(R_max_Rs, nfw_params['Rs'], nfw_params['alpha_Rs'])
        M_uldm = Uldm().mass_3d_lens(R_max_Rs, uldm_params['kappa_0'], uldm_params['theta_c'])

        factor = (M_nfw - M_uldm) / M_nfw
        nfw_params['alpha_Rs'] *= factor

        return nfw_params

class ULDMSubhalo(ULDMFieldHalo):
    """
    Defines a composite ULDM+NFW halo that is a subhalo of the host dark matter halo. The only difference
    between the profile classes is that the subhalo will have it's concentration evaluated at infall redshift.
    """
    @property
    def z_eval(self):
        """
        Returns the redshift at which to evaluate the concentration-mass relation
        """
        if not hasattr(self, '_zeval'):

            if self._args['evaluate_mc_at_zlens']:
                self._zeval = self.z
            else:
                self._zeval = self.z_infall

        return self._zeval
