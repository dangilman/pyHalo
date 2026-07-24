from scipy.optimize import brentq
from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.splcore import SPLCORE

class GlobularCluster(Halo):

    def __init__(self, mass, x, y, z, lens_cosmo_instance, args, unique_tag):
        """
        A cored steep-power-law (SPL_CORE) mass profile model for a globular cluster.

        :param mass: the total mass of the GC, defined as the mass enclosed within the
            radius gc_size_pc. For a steep outer slope (gamma >~ 5) essentially all the
            mass is inside gc_size_pc, so this equals the total profile mass to <0.1%.
        :param x: x coordinate [arcsec]
        :param y: y coordinate [arcsec]
        :param z: redshift
        :param lens_cosmo_instance: an instance of LensCosmo class
        :param args: keyword arguments for the profile, must contain 'gamma',
            'gc_size_pc', and 'gc_concentration'. gamma is the logarithmic density slope
            outside the core (rho ~ r^-gamma; use ~6 to mimic the sharp tidal truncation
            of a King profile). gc_size_pc is the GC size in parsecs (the mass is defined
            as the mass inside a sphere of this radius). gc_concentration is gc_size / r_core,
            the ratio of the size to the core radius.
        :param unique_tag:
        """
        self._prof = SPLCORE()
        self._lens_cosmo = lens_cosmo_instance
        mdef = 'SPL_CORE'
        is_subhalo = False
        super(GlobularCluster, self).__init__(mass, x, y, None, mdef, z, is_subhalo,
                                              lens_cosmo_instance, args, unique_tag,
                                              fixed_position=True)

    def density_profile_2d_lenstronomy(self, r):
        """

        :param r:
        :return:
        """
        rho0, gc_size, gamma, r_core = self.profile_args
        return self._prof.density_2d(r, 0.0, rho0, r_core, gamma)

    def density_profile_3d_lenstronomy(self, r):
        """

        :param r:
        :return:
        """
        kwargs = self.lenstronomy_params[0][0]
        kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
        sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
        sigma_crit_kpc = sigma_crit_mpc * 1e-6
        factor = sigma_crit_kpc / kpc_per_arcsec
        return factor * self._prof.density_lens(r / kpc_per_arcsec,
                                                kwargs['sigma0'],
                                                kwargs['r_core'],
                                                kwargs['gamma'])

    @property
    def half_mass_radius(self):
        """
        The 3D radius enclosing half of the mass (= half of self.mass, which is the mass
        within gc_size). Solved analytically from the enclosed-mass profile; the original
        implementation used a single np.trapz (a scalar), which made argmin trivial and
        returned r[0].
        :return: radius in pc
        """
        rho0, gc_size_kpc, gamma, r_core_kpc = self.profile_args
        r_half_kpc = brentq(lambda r: self._prof.mass_3d(r, rho0, r_core_kpc, gamma) - 0.5 * self.mass,
                            1e-4 * gc_size_kpc, gc_size_kpc)
        return r_half_kpc * 1000

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            gamma = self._args['gamma']
            gc_size_pc = self._args['gc_size_pc']
            gc_size_arcsec = (gc_size_pc * 1e-3) / kpc_per_arcsec
            r_core_arcsec = gc_size_arcsec / self._args['gc_concentration']
            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
            rho0 = self.mass / self._prof.mass_3d(gc_size_arcsec, sigma_crit_arcsec, r_core_arcsec, gamma)
            sigma0 = rho0 * r_core_arcsec
            self._lenstronomy_args = [{'sigma0': sigma0, 'gamma': gamma, 'center_x': self.x, 'center_y': self.y,
                                       'r_core': r_core_arcsec}]
        return self._lenstronomy_args, None

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            gamma = self._args['gamma']
            gc_size_pc = self._args['gc_size_pc']
            c = self._args['gc_concentration']
            gc_size_kpc = gc_size_pc * 1e-3
            r_core_kpc = gc_size_kpc / c
            rho0 = self.mass / self._prof.mass_3d(gc_size_kpc, 1.0, r_core_kpc, gamma)
            self._profile_args = (rho0, gc_size_kpc, gamma, r_core_kpc)
        return self._profile_args

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['SPL_CORE']

    @property
    def z_eval(self):
        """
        Returns the redshift at which to evalate the concentration-mass relation
        """
        return self.z
