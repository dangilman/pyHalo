from colossus.halo.concentration import *
from colossus.cosmology import cosmology
import astropy.cosmology as astropy_cosmo
from scipy.interpolate import interp1d
from pyHalo.defaults import *

cosmo_defaults = CosmoDefaults()

class Cosmology(object):

    M_sun = 1.98847 * 10 ** 30  # solar mass in [kg]

    Mpc = 3.08567758 * 10 ** 22  # Mpc in [m]

    arcsec = 2 * np.pi / 360 / 3600  # arc second in radian

    G = 6.67384 * 10 ** (-11) * Mpc**-3 * M_sun # Gravitational constant [Mpc^3 M_sun^-1 s^-2]

    c = 299792458*Mpc**-1 # speed of light [Mpc s^-1]

    density_to_MsunperMpc = 0.001 * M_sun**-1 * (100**3) * Mpc**3 # convert [g/cm^3] to [solarmasses / Mpc^3]

    def __init__(self, astropy_instance=None, cosmo_kwargs={}):

        self.astropy = self._setup_astropy_cosmology(astropy_instance, cosmo_kwargs)

        self._colossus_cosmo = self._setup_colossus_cosmology(cosmo_kwargs)

        self.h = self.astropy.h

        self._age_today = self.astropy.age(0).value

        self._DA_interp = self._interp_angular_diamter_distance()

        self._DC_interp = self._interp_comoving_distance()

        self._kpc_per_asec_interp = self._interp_kpc_per_asec()

    def D_A_z(self, z):

        try:
            return self._DA_interp(z)
        except:
            return self.D_A(0, z)

    def D_C_z(self, z):

        try:
            return self._DC_interp(z)
        except:
            return self.D_A_z(z) / self.scale_factor(z)

    def kpc_proper_per_asec(self, z):

        return self._kpc_per_asec_interp(z)

    @property
    def colossus(self):
        return self._colossus_cosmo

    def halo_age(self, z, zform=10):

        halo_form = self.astropy.age(zform).value
        if z > zform:
            return 0
        else:
            return self.astropy.age(z).value - halo_form

    def scale_factor(self, z):

        return (1 + z) ** -1

    def D_A(self, z1, z2):

        return self.astropy.angular_diameter_distance_z1z2(z1, z2).value

    def D_C_transverse(self, z):

        return self.astropy.comoving_transverse_distance(z).value

    def E_z(self, z):

        return self.astropy.efunc(z)

    def rho_crit(self, z):
        """
        :param z: redshift
        :return: critical density of the universe at redshift z in solar mass / Mpc^3
        """

        return self.astropy.critical_density(z).value * self.density_to_MsunperMpc

    @property
    def rho_dark_matter_crit(self):
        """
        :param z: redshift
        :return: comoving dark matter density of the universe at redshift z in solar mass / Mpc^3
        """

        return self.astropy.Odm(0.) * self.rho_crit(0.)

    def _interp_kpc_per_asec(self):

        zmin, zmax = 0.001, 4
        z = np.arange(zmin, zmax + 0.025, 0.025)
        kpc_per_asec = [self.astropy.arcsec_per_kpc_proper(zi).value ** -1 for zi in z]
        kpc_per_asec = np.array(kpc_per_asec)

        return interp1d(z, kpc_per_asec)

    def _interp_angular_diamter_distance(self):

        zmax = 4
        zstep = lenscone_default.default_z_step
        z = np.arange(zstep, zmax+zstep, zstep)
        da = []
        for zi in z:
            da.append(self.D_A(0, zi))
        da = np.array(da)
        return interp1d(z, da)

    def _interp_comoving_distance(self):

        zmax = 4
        zstep = lenscone_default.default_z_step
        z = np.arange(zstep, zmax + zstep, zstep)
        dc = []
        for zi in z:
            dc.append(self.D_C_transverse(zi))
        dc = np.array(dc)
        return interp1d(z, dc)

    def _setup_astropy_cosmology(self, astropy_instance, cosmo_kwargs):

        if not hasattr(self, 'astropy'):

            if astropy_instance is None:

                astropy_kwargs = {}

                keys = ['H0', 'Om0', 'Ob0']
                for key in keys:
                    if key not in cosmo_kwargs.keys():
                        astropy_kwargs.update({key: cosmo_defaults(key)})
                    else:
                        astropy_kwargs.update({key: cosmo_kwargs[key]})

                astropy_instance = astropy_cosmo.FlatLambdaCDM(**astropy_kwargs)

            self.astropy = astropy_instance

        return self.astropy

    def _setup_colossus_cosmology(self, cosmo_kwargs):

        if not hasattr(self,'colossus_cosmo'):

            colossus_kwargs = {}
            keys = ['H0', 'Om0', 'Ob0', 'ns', 'sigma8', 'power_law']
            for key in keys:
                if key not in cosmo_kwargs.keys():
                    colossus_kwargs.update({key: cosmo_defaults(key)})
                else:
                    colossus_kwargs.update({key: cosmo_kwargs[key]})
            self._colossus_cosmo = cosmology.setCosmology('custom', colossus_kwargs)

        return self._colossus_cosmo
