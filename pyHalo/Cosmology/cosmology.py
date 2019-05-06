from scipy.integrate import quad
from colossus.halo.concentration import *
from colossus.cosmology import cosmology
import astropy.cosmology as astropy_cosmo
from scipy.special import hyp2f1

class Cosmology(object):

    M_sun = 1.9891 * 10 ** 30  # solar mass in [kg]

    Mpc = 3.08567758 * 10 ** 22  # Mpc in [m]

    arcsec = 2 * np.pi / 360 / 3600  # arc second in radian

    G = 6.67384 * 10 ** (-11) * Mpc**-3 * M_sun # Gravitational constant [Mpc^3 M_sun^-1 s^-2]

    c = 299792458*Mpc**-1 # speed of light [Mpc s^-1]

    density_to_MsunperMpc = 0.001 * M_sun**-1 * (100**3) * Mpc**3 # convert [g/cm^3] to [solarmasses / Mpc^3]

    def __init__(self, rescale_sigma8 = False, H0=69.7, omega_baryon = 0.0464, omega_DM = 0.235, sigma8 = 0.82, **kwargs):

        self.astropy = astropy_cosmo.FlatLambdaCDM(H0=H0, Om0=omega_baryon + omega_DM, Ob0=omega_baryon)

        if rescale_sigma8 is True:
            self.sigma_8 = self.rescale_sigma8(sigma8,void_omega_M=kwargs['omega_M_void'])
            self.delta_c = 1.62 # appropriate for an "open" universe in the void
        else:
            self.sigma_8 = sigma8
            self.delta_c = 1.68647 # appropriate for a flat universe

        self.h = self.astropy.h

        self._colossus_cosmo = self._set_colossus_cosmo()

        self._age_today = self.astropy.age(0).value

    def _set_colossus_cosmo(self):

        if not hasattr(self,'colossus_cosmo'):

            # self._cosmology_params = {'omega_M_0':self.cosmo.Om0, 'omega_b_0':self.cosmo.Ob0, 'omega_lambda_0': 1 - self.cosmo.Om0,
            #                  'omega_n_0':0,'N_nu':1,'h':self.h,'sigma_8':self.sigma_8,'n':1}

            params = {'flat': True, 'H0': self.h*100, 'Om0':self.astropy.Om0,
                      'Ob0':self.astropy.Ob0, 'sigma8':self.sigma_8, 'ns': 0.9608}

            self._colossus_cosmo = cosmology.setCosmology('custom', params)

        return self._colossus_cosmo

    def halo_age(self, z, zform=10):

        halo_form = self.astropy.age(zform).value
        if z > zform:
            return 0
        else:
            return self.astropy.age(z).value - halo_form

    def scale_factor(self,z):

        return (1+z)**-1

    def D_A(self,z1,z2):

        return self.astropy.angular_diameter_distance_z1z2(z1, z2).value

    def D_C(self,z):

        return self.astropy.comoving_distance(z).value

    def E_z(self,z):

        a = self.a_z(z)

        return np.sqrt(self.astropy.Om0 * a ** -3 + (1 - self.astropy.Om0))

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        """
        return 1. / (1 + z)

    def radian_to_asec(self,x):
        """

        :param x: angle in radians
        :return:
        """
        return x*self.arcsec**-1

    def kpc_per_asec(self,z):

        return self.astropy.arcsec_per_kpc_proper(z).value ** -1

    def D_ratio(self,z1,z2):

        return self.D_A(z1[0],z1[1])*self.D_A(z2[0],z2[1])**-1

    def f_z(self,z):

        I = quad(self.E_z,0,z)[0]
        I2 = quad(self.E_z,0,self.zd)[0]

        return (1+z)**-2*self.E_z(z)**-1*(I**2*I2**-2)

    def T_xy(self, z_observer, z_source):
        """
        transverse comoving distance in units of Mpc
        """
        T_xy = self.astropy.comoving_transverse_distance(z_source).value - self.astropy.comoving_transverse_distance(z_observer).value

        return T_xy

    def D_xy(self, z_observer, z_source):
        """
        angular diamter distance in units of Mpc
        :param z_observer: observer
        :param z_source: source
        :return:
        """
        a_S = self.a_z(z_source)
        D_xy = (self.astropy.comoving_transverse_distance(z_source) - self.astropy.comoving_transverse_distance(z_observer)) * a_S
        return D_xy.value

    def rho_crit(self,z):
        """
        :param z: redshift
        :return: critical density of the universe at redshift z in solar mass / Mpc^3
        """
        return self.astropy.critical_density(z).value * self.density_to_MsunperMpc

    def rho_matter_crit(self,z):
        """
        :param z: redshift
        :return: matter density of the universe at redshift z in solar mass / Mpc^3
        """

        return self.rho_crit(z)*self.astropy.Om(z)

    def D_growth(self,z,omega_M,omega_L):

        def f(x,OmO,OmL):
            return (1+OmO*(x**-1 - 1)+OmL*(x**2-1))**-.5

        a = (1+z)**-1

        if omega_M+omega_L != 1:
            return a * hyp2f1(3 ** -1, 1, 11 * 6 ** -1, a ** 3 * (1 - omega_M ** -1))
        else:
            prefactor = 5*omega_M*(2*a*f(a,omega_M,omega_L))**-1
            return prefactor*quad(f,0,a,args=(omega_M,omega_L))[0]

    def rescale_sigma8(self,sigma_8_init,void_omega_M):

        """
        :param sigma_8_init: initial cosmological sigma8 in the field
        :param void_omega_M: the matter density in the void
        :return: a rescaled sigma8 appropriate for an under dense region
        Gottlober et al 2003
        """

        zi = 1000

        D_ai = self.D_growth(zi, self.astropy.Om0, self.astropy.Ode0)
        D_a1 = self.D_growth(0, self.astropy.Om0, self.astropy.Ode0)

        D_void_ai = self.D_growth(zi, void_omega_M, self.astropy.Ode0)
        D_void_a1 = self.D_growth(0, void_omega_M, self.astropy.Ode0)

        return sigma_8_init*(D_ai*D_void_a1)*(D_a1*D_void_ai)**-1

