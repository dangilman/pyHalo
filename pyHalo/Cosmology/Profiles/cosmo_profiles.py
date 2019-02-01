import numpy
from colossus.halo.concentration import concentration

class CosmoMassProfiles(object):

    def __init__(self, lens_comso):

        self.lens_cosmo = lens_comso

    def rN_M_nfw(self, M, N):
        """
        computes the radius R_N of a halo of mass M in comoving distances M/h
        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :return: radius R_200 in comoving Mpc/h
        """

        return (3 * M / (4 * numpy.pi * self.lens_cosmo.rhoc * N)) ** (1. / 3.)

    def NFW_concentration(self,M,z,model='diemer18',mdef='200c',logmhm=0,
                                scatter=True,g1=None,g2=None):

        # WDM relation adopted from Ludlow et al
        # use diemer18?
        def zfunc(z_val):
            return 0.026*z_val - 0.04

        if isinstance(M, float) or isinstance(M, int):
            c = concentration(M*self.cosmo.h,mdef=mdef,model=model,z=z)
        else:
            con = []
            for i,mi in enumerate(M):

                con.append(concentration(mi*self.cosmo.h,mdef=mdef,model=model,z=z[i]))
            c = numpy.array(con)

        if logmhm != 0:

            mhm = 10**logmhm
            concentration_factor = (1+g1*mhm*M**-1)**g2
            redshift_factor = (1+z)**zfunc(z)
            rescale = redshift_factor * concentration_factor

            c = c * rescale

        # scatter from Dutton, maccio et al 2014
        if scatter:

            if isinstance(c, float) or isinstance(c, int):
                c = numpy.random.lognormal(numpy.log(c),0.13)
            else:
                con = []
                for i, ci in enumerate(c):
                    con.append(numpy.random.lognormal(numpy.log(ci),0.13))
                c = numpy.array(con)
        return c

    def truncation_roche(self, M, r3d, k = 0.68, nu = 2):

        """

        :param M: m200
        :param r3d: 3d radial position in the halo (kpc)
        :return: Equation 2 in Gilman et al 2019 (expressed in arcsec)
        """

        exponent = nu * 3 ** -1
        rtrunc_kpc = k*(M / 10**6) ** (1./3) * (r3d * 100 ** -1) ** (exponent)

        return rtrunc_kpc * self.lens_cosmo._kpc_per_asec_zlens ** -1
