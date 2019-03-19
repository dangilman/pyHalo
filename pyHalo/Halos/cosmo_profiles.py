import numpy
from colossus.halo.concentration import concentration
from scipy.optimize import minimize

class CosmoMassProfiles(object):

    def __init__(self, lens_comso = None, z_lens = None, z_source = None):

        if lens_comso is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            assert z_lens is not None
            assert z_source is not None
            lens_comso = LensCosmo(z_lens, z_source)

        self.lens_cosmo = lens_comso

    def rho0_c_NFW(self, c):
        """
        computes density normalization as a function of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """

        return 200. / 3 * self.lens_cosmo.rhoc * c ** 3 / (numpy.log(1 + c) - c / (1 + c))

    def _nfwParam_physical_Mpc(self, M, c, z):

        """
        returns the NFW parameters in physical units
        :param M: physical mass in M_sun
        :param c: concentration
        :return:
        """

        h = self.lens_cosmo.cosmo.h
        a_z = self.lens_cosmo.cosmo.scale_factor(z)

        r200 = self.rN_M_nfw_comoving(M * h, 200) * a_z / h   # physical radius r200
        rho0 = self.rho0_c_NFW(c) * h ** 2 / a_z ** 3 # physical density in M_sun/Mpc**3

        Rs = r200/c
        return rho0, Rs, r200

    def NFW_params_physical(self, M, c, z):

        """

        :param M: physical M200
        :param c: concentration
        :param z: halo redshift
        :return: physical NFW parameters in kpc units
        """
        rho0, Rs, r200 = self._nfwParam_physical_Mpc(M, c, z)

        return rho0 * 1000 ** -3, Rs * 1000, r200 * 1000

    def rN_M_nfw_comoving(self, M, N):
        """
        computes the radius R_N of a halo of mass M in comoving distances
        :param M: halo mass in M_sun
        :type M: float or numpy array
        :return: radius R_200 in comoving Mpc/h
        """

        return (3 * M / (4 * numpy.pi * self.lens_cosmo.rhoc * N)) ** (1. / 3.)

    def rN_M_nfw_physical(self, M, N, z):
        """
        computes the radius R_N of a halo of mass M in physical distance
        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :return: radius R_200 in physical Mpc
        """

        a_z = (1 + z) ** -1

        h = self.lens_cosmo.cosmo.h

        rnm_comoving = self.rN_M_nfw_comoving(M * h, N)

        return rnm_comoving * a_z * h ** -1

    def rN_M_nfw_physical_arcsec(self, M, N, z):

        rNM_physical = self.rN_M_nfw_physical(M, N, z)

        rNM_arcsec = rNM_physical * 1000 * self.lens_cosmo.cosmo.kpc_per_asec(z) ** -1

        return rNM_arcsec

    def NFW_concentration(self, M, z, model='diemer18', mdef='200c', logmhm=0,
                          scatter=True, c_scale=None, c_power=None):

        # WDM relation adopted from Ludlow et al
        # use diemer18?
        def zfunc(z_val):
            return 0.026*z_val - 0.04

        if isinstance(M, float) or isinstance(M, int):
            c = concentration(M*self.lens_cosmo.cosmo.h,mdef=mdef,model=model,z=z)
        else:
            con = []
            for i,mi in enumerate(M):

                con.append(concentration(mi*self.lens_cosmo.cosmo.h,mdef=mdef,model=model,z=z[i]))
            c = numpy.array(con)

        if logmhm != 0:

            mhm = 10**logmhm
            concentration_factor = (1 + c_scale * mhm * M ** -1) ** c_power
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

    def truncation_roche(self, M, r3d, z, k, nu):

        """

        :param M: m200
        :param r3d: 3d radial position in the halo (kpc)
        :return: Equation 2 in Gilman et al 2019 (expressed in arcsec)
        (k tuned to match output of truncation roche exact)
        """

        exponent = nu * 3 ** -1
        rtrunc_kpc = k*(M / 10**6) ** (1./3) * (r3d * 100 ** -1) ** (exponent)

        return rtrunc_kpc * self.lens_cosmo.cosmo.kpc_per_asec(z) ** -1

    def truncation_roche_isonfw(self, msub, r3d, m_parent, z, logmhm=0, g1=None, g2 = None, k=1,
                                m_func = 'NFW', beta_sub= 1e-5):

        c_parent = self.NFW_concentration(m_parent, z, logmhm=logmhm,
                                          scatter=False, c_scale=g1, c_power=g2)
        c_sub = self.NFW_concentration(msub, z, logmhm=logmhm,
                                       scatter=False, c_scale=g1, c_power=g2)

        rho_sub, Rs_sub, _ = self.NFW_params_physical(msub, c_sub, z)
        _, Rs_main, _ = self.NFW_params_physical(m_parent, c_parent, z)

        fc_parent = numpy.log(1+r3d * Rs_main**-1)

        area = (numpy.pi * 1) ** 2
        m_main = self.lens_cosmo.sigmacrit * area
        rho_main = m_main * (4 * numpy.pi * Rs_main ** 3 * fc_parent) ** -1

        if m_func == 'NFW':
            fc = numpy.log(c_sub + 1) - c_sub * (1+c_sub) ** -1
        elif m_func == 'coreNFW':
            fc = (c_sub * (1+c_sub) ** -1 * (-1+beta_sub) ** -1 + (-1+beta_sub) ** -2 *
                      ((2*beta_sub-1)*numpy.log(1/(1+c_sub)) + beta_sub **2 * numpy.log(c_sub / beta_sub + 1)))

        rho_factor = (rho_sub * fc) * (fc_parent * rho_main) ** -1
        rs_factor = Rs_sub * Rs_main ** -1
        r_trunc = k * r3d * rs_factor * rho_factor ** (1./3)

        return r_trunc, r_trunc * Rs_sub ** -1

    def truncation_roche_exact(self, msub, r3d, m_parent, z, logmhm=0,
                               g1=None,g2=None):

        """
        explict evaluation of the Roche radius Tormen et al. 1998
        interpreted as mean(rho_satellite(r_t)) = mean(rho_parent(r_3d))

        :param msub: subhalo mass
        :param r3d: subhalo 3d position
        :param m_parent: parent halo mass (assuing NFW profile)
        :param z: redshift
        :param logmhm, g1, g2: half mode mass, concentration parameters
        :return: the Roche radius in units of r3d
        """

        def _fx(y):
            return numpy.log(1+y) - y * (1+y) ** -1
        def _f1(t):
            return t**3 * _fx(t)**-1
        def _f2(x):
            return _f1(x) * gamma
        def _func_to_minimize(tau):
            return numpy.absolute(_f1(tau) - _f2(X))

        c_parent = self.NFW_concentration(m_parent, z, logmhm=logmhm,
                                          scatter=False, c_scale=g1, c_power=g2)
        c_sub = self.NFW_concentration(msub, z, logmhm=logmhm,
                                       scatter=False, c_scale=g1, c_power=g2)

        rho_sub, Rs_sub, _ = self.NFW_params_physical(msub, c_sub, z)
        rho_main, Rs_main, _ = self.NFW_params_physical(m_parent, c_parent, z)

        gamma = (c_sub * c_parent**-1) ** 3 * (_fx(c_sub) * _fx(c_parent)**-1) * \
            (rho_main * rho_sub**-1)
        X = r3d * Rs_main ** -1

        opt = minimize(_func_to_minimize, x0=[5], method='Nelder-Mead')
        tau = float(opt['x'])

        if tau < 1:
            tau = 1

        return tau * Rs_sub

    def LOS_truncation(self, M, z, N=50):
        """
        Truncate LOS halos at r50
        :param M:
        :param c:
        :param z:
        :param N:
        :return:
        """

        r_trunc_arcsec = self.rN_M_nfw_physical_arcsec(M, N, z)

        return r_trunc_arcsec
