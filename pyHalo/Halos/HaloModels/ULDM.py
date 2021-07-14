from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.concentration import Concentration
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.cnfw import CNFW
from lenstronomy.LensModel.Profiles.uldm import Uldm 
from scipy.optimize import minimize
import lenstronomy.Util.constants as const
from pyHalo.Halos.lens_cosmo import LensCosmo
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

        return [self._args['nfw_mdef'], 'ULDM']

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
        if self._args['nfw_mdef'] == 'TNFW':

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
            kwargs_nfw = self._rescaled_tnfw_params(kwargs_nfw_temporary, kwargs_uldm)
            kwargs = [kwargs_nfw, kwargs_uldm]

            return kwargs, None
        
        elif self._args['nfw_mdef'] == 'CNFW':

            [concentration, theta_c, kappa_0] = self.profile_args
            Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)

            x, y = np.round(self.x, 4), np.round(self.y, 4)

            Rs_angle = np.round(Rs_angle, 10)
            theta_Rs = np.round(theta_Rs, 10)

            kwargs_nfw_temporary = {'alpha_Rs': theta_Rs, 'Rs': Rs_angle,
                'center_x': x, 'center_y': y, 'r_core': theta_c}

            kwargs_uldm_temporary = {'theta_c': theta_c, 'kappa_0': kappa_0,
                'center_x': x, 'center_y': y}

            kwargs = self._rescaled_cnfw_params(kwargs_nfw_temporary,
                                                kwargs_uldm_temporary)

            return kwargs, None
        
        else:
            raise ValueError('Choose TNFW (truncated) or CNFW (cored) for nfw_type.')



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

            if self._args['nfw_mdef'] == 'TNFW':

                truncation_radius = self._lens_cosmo.LOS_truncation_rN(self.mass, self.z,
                                                             self._args['LOS_truncation_factor'])

                # using 'mass' as the ULDM virial mass
                [theta_c, kappa_0] = self._uldm_args(self._args['log10_m_uldm'], self.mass, self._args['uldm_plaw'])

                self._profile_args = (self.c, truncation_radius, theta_c, kappa_0)
            
            elif self._args['nfw_mdef'] == 'CNFW':

                [theta_c, kappa_0] = self._uldm_args(self._args['log10_m_uldm'], self.mass, self._args['uldm_plaw'])
                
                self._profile_args = (self.c, theta_c, kappa_0)

        return self._profile_args
    
    def _uldm_args(self, m_log10, M, plaw):
        """
        :param m_log10: ULDM particle mass in log10 units
        :param M: ULDM virial mass, same as TNFW virial mass 
        :param plaw: power law exponent for core radius - virial mass relationship

        :return: core radius 'theta_c' and core density 'kappa_0' in lensing units.
        """
        r_c = self._core_radius(m_log10, M, plaw)
        rho_c = self._central_density(m_log10, r_c)

        Sigma_crit = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source) * 1e-12
        D_lens = self._lens_cosmo.D_d * 1e3

        theta_c = r_c / D_lens / const.arcsec 
        kappa_0 = 429 * np.pi * rho_c * (r_c * 1e3) / (2048 * np.sqrt(0.091) * Sigma_crit)

        return [theta_c, kappa_0]
    
    def _core_radius(self, m_log10, M, plaw):
        """
        :param m_log10: ULDM particle mass in log10 units
        :param M: ULDM virial mass, same as TNFW virial mass 
        :param plaw: power law exponent for core radius - virial mass relationship

        :return: core radius in kpc, numerator of equation (3) in Schive et al. 2014 [1407.7762v2]
        """
        m22 = 10**(m_log10 + 22)
        a = 1/(1+self.z)
        M9 = 10**(np.log10(M) - 9)
        Zeta = (self._zeta(self.z) / self._zeta(0))

        return 1.6 * m22**(-1) * a**(1/2) * Zeta**(-1/6) * M9**(-plaw)
    
    def _central_density(self, m_log10, r_c):
        """
        :param m_log10: ULDM particle mass in log10 units
        :param r_c: core radius in kpc

        :return: central density in M_solar/pc^3, equation (7) in Schive et al. 2014 [1407.7762v2]
        """
        m23 = 10**(m_log10 + 23)
        a = 1/(1+self.z)
        x_c = r_c/a

        return 1.9 * a**(-1) * m23**(-2) * x_c**(-4)

    def _rescaled_tnfw_params(self, tnfw_params, uldm_params):
        """
        :param tnfw_params: truncated NFW halo lensing params
        :param uldm_params: ULDM halo lensing params

        :return: rescaled truncated NFW params to fill up the remainder of the mass budget such that
        the composite profile has the inputted virial mass.
        """
        r200 = self.c * tnfw_params['Rs'] 
        rho0 = tnfw_params['alpha_Rs'] / (4. * tnfw_params['Rs'] ** 2 * (1. + np.log(1. / 2.)))
        M_nfw = TNFW().mass_3d(r200, tnfw_params['Rs'], rho0, tnfw_params['r_trunc'])
        M_uldm = Uldm().mass_3d(r200, uldm_params['kappa_0'], uldm_params['theta_c'])

        factor = (M_nfw - M_uldm) / M_nfw

        if factor < 0:
            raise ValueError('Negative NFW profile mass, tweaky your parameters.')
        else:
            pass

        tnfw_params['alpha_Rs'] *= factor
        return tnfw_params
    
    def _rescaled_cnfw_params(self, cnfw_params, uldm_params):
        """
        :param tnfw_params: cored NFW halo lensing params
        :param uldm_params: ULDM halo lensing params

        :return: rescaled cored NFW params to fill up the remainder of the mass budget such that
        the composite profile has the inputted virial mass.
        """
        r200 = self.c * cnfw_params['Rs'] 
        initial_guess = np.array([0.5,0.5])
        rho0 = Uldm().density_lens(0,uldm_params['kappa_0'],
                                    uldm_params['theta_c'])
        rhos = CNFW().density_lens(0,cnfw_params['Rs'],
                                    cnfw_params['alpha_Rs'],
                                    cnfw_params['r_core'])

        args = (r200, self.mass, cnfw_params['Rs'], cnfw_params['alpha_Rs'], 
                        uldm_params['kappa_0'], uldm_params['theta_c'],
                        rho0, rhos)
        
        beta,q = minimize(self._function_to_minimize, initial_guess, 
                                args, method='Nelder-Mead')['x']

        cnfw_params['r_core'] *= beta
        uldm_params['kappa_0'] *= q

        return [cnfw_params, uldm_params]
    
    def _zeta(self,z):
        Om_z = self._lens_cosmo.cosmo.astropy.Om(z)
        return (18*np.pi**2 + 82*(Om_z-1) - 39*(Om_z-1)**2) / Om_z
    
    def _constraint_mass(self, beta, q, r, m_target, rs, alpha_rs, kappa_0, theta_c):
        """
        Evaluate the mass constraint equation for CNFW component profile
        """
        r_core = beta * rs
        args_nfw = (r, rs, alpha_rs, r_core)
        args_uldm = (r, kappa_0, theta_c)
        
        m_nfw = CNFW().mass_3d_lens(*args_nfw) / m_target
        m_uldm = q * Uldm().mass_3d_lens(*args_uldm) / m_target
        
        # penalize if not equal to zero
        return np.absolute(m_nfw + m_uldm - 1)
    
    def _constraint_density(self, beta, q, rho_target, rhos):
        """
        Evaluate the density constraint equation for CNFW component profile
        """
        
        # penalize if not equal to zero
        return np.absolute(rhos - beta * rho_target * (q - 1))
    
    def _function_to_minimize(self, beta_q_args, r, m_target, rs, alpha_rs, kappa_0, theta_c, rho0, rhos):
        """
        Add the two constraints for CNFW component profile
        """
        
        # minimize will work with an array of arguments, so need to pass in an array and unpack it
        (beta, q) = beta_q_args
        
        constraint1 = self._constraint_mass(beta, q, r, m_target, rs, alpha_rs, kappa_0, theta_c)
        constraint2 = self._constraint_density(beta, q, rho0, rhos)
        return constraint1 + constraint2

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
