from pyHalo.Halos.cosmo_profiles import CosmoMassProfiles


class PTmass(CosmoMassProfiles):

    def point_mass_fac(self,z):
        """
        This factor times sqrt(M) gives the einstein radius for a point mass in arcseconds
        :param z:
        :return:
        """
        D_d = self.lens_cosmo.cosmo.D_A(0, z)
        D_s = self.lens_cosmo.cosmo.D_A(0, self.lens_cosmo.z_source)
        D_ds = self.lens_cosmo.cosmo.D_A(z, self.lens_cosmo.z_source)

        const = 4 * self.lens_cosmo.cosmo.G * self.lens_cosmo.cosmo.c ** -2 * D_ds * (D_d * D_s) ** -1

        return self.lens_cosmo.cosmo.arcsec ** -1 * const ** .5

