from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.cnfw import CNFW
from lenstronomy.LensModel.Profiles.uldm import Uldm
from scipy.optimize import minimize
import lenstronomy.Util.constants as const
import numpy as np

class ULDMFieldHalo(Halo):

    """
    The base class for a truncated NFW halo
    """
    def __init__(self, mass, x, y, r3d, z,
                 sub_flag, lens_cosmo_instance, args, truncation_class, concentration_class, unique_tag):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        self._truncation_class = truncation_class
        self._concentration_class = concentration_class
        mdef = 'ULDM'
        super(ULDMFieldHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag,
                                            lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['CNFW', 'ULDM']

    @property
    def c(self):
        """
        Computes the NFW halo concentration (once)
        """
        if not hasattr(self, '_c'):
            self._c = self._concentration_class.nfw_concentration(self.mass, self.z_eval)
        return self._c

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        [concentration, theta_c, kappa_0] = self.profile_args

        x, y = np.round(self.x, 4), np.round(self.y, 4)
        Rs_angle, alpha_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)
        kwargs_cnfw_temporary = {'alpha_Rs': alpha_Rs, 'Rs': Rs_angle,
            'center_x': x, 'center_y': y, 'r_core': 3*theta_c}
        kwargs_uldm_temporary = {'theta_c': theta_c, 'kappa_0': kappa_0,
            'center_x': x, 'center_y': y}
        kwargs = self._rescaled_cnfw_params(kwargs_cnfw_temporary,
                                                kwargs_uldm_temporary)
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

            [theta_c, kappa_0] = self._uldm_args(self._args['log10_m_uldm'], self.mass, self._args['uldm_plaw'])
            self._profile_args = (self.c, theta_c, kappa_0)

        return self._profile_args

    def _uldm_args(self, m_log10, M, plaw):
        """
        :param m_log10: ULDM particle mass in log10 units
        :param M: ULDM virial mass
        :param plaw: power law exponent for core radius - virial mass relationship

        :return: core radius 'theta_c' and core density 'kappa_0' in lensing units.
        """
        r_c = self._core_radius(m_log10, M, plaw) # in kpc
        rho_c = self._central_density(m_log10, r_c) # in M_solar/pc^3
        r_c *= 10**3 # in pc
        Sigma_crit = self._lens_cosmo.sigma_crit_lensing * 10**(-12) # in M_solar/parsec^2
        D_lens = self._lens_cosmo.D_d * 10**6 # in pc
        theta_c = r_c / D_lens / const.arcsec # in arcsec
        kappa_0 = 429 * np.pi * rho_c * r_c / (2048 * np.sqrt(0.091) * Sigma_crit) # lensing units
        return [theta_c, kappa_0]

    def _core_radius(self, m_log10, M, plaw):
        """
        :param m_log10: ULDM particle mass in log10 units
        :param M: ULDM virial mass
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

    def _zeta(self,z):
        """
        :param z: redshift at which to evalute function

        :return: zeta function at z, see definition of ULDM density
        """
        Om_z = self._lens_cosmo.cosmo.astropy.Om(z)
        return (18*np.pi**2 + 82*(Om_z-1) - 39*(Om_z-1)**2) / Om_z

    def _rescaled_cnfw_params(self, cnfw_params, uldm_params):
        """
        :param cnfw_params: cored NFW halo lensing params
        :param uldm_params: ULDM halo lensing params

        :return: rescaled cored NFW params to fill up the remainder of the mass budget such that
        the composite profile has the inputted virial mass.
        """

        r200 = self._c * cnfw_params['Rs']
        rho0 = Uldm().density_lens(0,uldm_params['kappa_0'],
                                    uldm_params['theta_c'])
        rhos = CNFW().density_lens(0,cnfw_params['Rs'],
                                 cnfw_params['alpha_Rs'],
                                 cnfw_params['r_core'])

        args = (r200, self.mass, cnfw_params['Rs'], cnfw_params['alpha_Rs'],
                        uldm_params['kappa_0'], uldm_params['theta_c'],
                        rho0, rhos)
        initial_guess = np.array([0.9,1.1])
        bounds = ((0.5, 10), (0.5, 1.5))
        method = 'Nelder-Mead'
        out = minimize(self._function_to_minimize, initial_guess,
                           args, method=method, bounds=bounds, tol=0.001)
        beta, q = out['x']

        if beta<0:
            raise ValueError('Negative CNFW core radius, tweak your parameters.')
        elif q<0:
            raise ValueError('Negative ULDM profile mass, tweak your parameters.')
        else:
            pass

        cnfw_params['r_core'] /= beta
        uldm_params['kappa_0'] /= q

        M_nfw = CNFW().mass_3d_lens(r200, cnfw_params['Rs'],
                    cnfw_params['alpha_Rs']*self._lens_cosmo.sigmacrit, cnfw_params['r_core'])
        M_uldm = Uldm().mass_3d_lens(r200,
                    uldm_params['kappa_0']*self._lens_cosmo.sigmacrit, uldm_params['theta_c'])

        if (self._args['scale_nfw']):
            # When scale_nfw is True rescale alpha_Rs to improve mass accuracy
            scale = self.mass / (M_nfw+M_uldm)
            cnfw_params['alpha_Rs'] *= scale

        return [cnfw_params, uldm_params]

    def _constraint_mass(self, beta, q, r, m_target, rs, alpha_rs, kappa_0, theta_c):
        """
        :param beta: CNFW core radius ('r_core') rescaling parameter
        :param q: ULDM core density ('kappa_0') rescaling parameter
        :param r: r200 of CNFW profile
        :param m_target: halo virial mass
        :param rs: CNFW scale radius
        :param alpha_rs: CNFW deflection angle at rs, in absence of core
        :param kappa_0: ULDM core density
        :param theta_c: ULDM core radius

        :return: Evaluated mass constraint equation for CNFW component profile
        """
        r_core = beta * rs
        sigma_crit = self.lens_cosmo.sigmacrit
        args_nfw = (r, rs, alpha_rs*sigma_crit, r_core)
        args_uldm = (r, kappa_0*sigma_crit, theta_c)

        m_nfw = CNFW().mass_3d_lens(*args_nfw) / m_target
        m_uldm = q * Uldm().mass_3d_lens(*args_uldm) / m_target

        penalty = np.absolute(m_nfw + m_uldm - 1)
        if np.isnan(penalty):
            return 1e+12

        # penalize if not equal to zero
        return penalty

    def _constraint_density(self, beta, q, rho_target, rhos):
        """
        :param beta: CNFW core radius ('r_core') rescaling parameter
        :param q: ULDM core density ('kappa_0') rescaling parameter
        :param rho_target: ULDM density at r=0
        :param rhos: CNFW density at r=0

        :return: Evaluated density constraint equation for CNFW component profile
        """

        # penalize if not equal to zero
        return np.absolute(rhos - beta * rho_target * (q - 1))

    def _function_to_minimize(self, beta_q_args, r, m_target, rs, alpha_rs, kappa_0, theta_c, rho0, rhos):
        """
        :param beta_q_args: array containing beta, q parameters, see _constraint_mass and _constraint_density
        :param r: r200 of CNFW profile
        :param m_target: halo virial mass
        :param rs: CNFW scale radius
        :param alpha_rs: CNFW deflection angle at rs, in absence of core
        :param kappa_0: ULDM core density
        :param theta_c: ULDM core radius
        :param rho0: ULDM density at r=0
        :param rhos: CNFW density at r=0

        :return: Addition of mass and density constraints for CNFW component profile
        """

        # minimize will work with an array of arguments, so need to pass in an array and unpack it
        (beta, q) = beta_q_args

        constraint1 = self._constraint_mass(beta, q, r, m_target, rs, alpha_rs, kappa_0, theta_c)
        constraint2 = self._constraint_density(beta, q, rho0, rhos)

        return constraint1 + 20*constraint2

class ULDMSubhalo(ULDMFieldHalo):
    """
    Defines a composite ULDM+NFW halo that is a subhalo of the host dark matter halo. The only difference
    between the profile classes is that the subhalo will have it's concentration evaluated at infall redshift.
    """
    @property
    def z_eval(self):
        """
        Returns the redshift at which to evalate the concentration-mass relation
        """
        if not hasattr(self, '_zeval'):

            if 'evaluate_mc_at_zlens' in self._args.keys() and self._args['evaluate_mc_at_zlens']:
                self._zeval = self.z
            else:
                self._zeval = self.z_infall

        return self._zeval
