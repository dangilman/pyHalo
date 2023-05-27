import numpy as np
import pickle
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from scipy.interpolate import RegularGridInterpolator
from pyHalo.Halos.util import tau_mf_interpolation
from colossus.lss import peaks
from colossus.halo import splashback

class TruncationSplashBack(object):

    def __init__(self, lens_cosmo):
        """
        This computes the splashback radius of a halo as the truncation radius (appropriate for field halos)
        See Diemer (2020)
        :param lens_cosmo: an instance of LensCosmo
        """
        self._lens_cosmo = lens_cosmo

    def truncation_radius_halo(self, halo):
        """
        Thiis method computes the truncation radius using the class attributes of an instance of Halo
        :param halo: an instance of halo
        :return: the truncation radius in physical kpc
        """
        return self.truncation_radius(halo.mass, halo.z)

    def truncation_radius(self, halo_mass, z):
        """
        Computes the radius r_N of an NFW halo
        :param halo_mass: halo mass (m200 with respect to critical density at z)
        :param z: redshift
        :param halo_concentration: the halo concentration
        :param lens_cosmo: an instance of LensCosmo
        :return: the truncation radius
        """
        h = self._lens_cosmo.cosmo.h
        nu200m = peaks.peakHeight(halo_mass * h, z)
        RspR200m_units_rvir, _ = splashback.splashbackModel('RspR200m', nu200m=nu200m, z=z, rspdef='sp-apr-mn')
        r200m_mpc = self._lens_cosmo.rN_M(halo_mass * h, z, 200.0) / self._lens_cosmo.cosmo.astropy.Om0 ** (1./3)
        rt_kpc = RspR200m_units_rvir * r200m_mpc * 1e3
        return rt_kpc

class TruncationRN(object):

    def __init__(self, lens_cosmo, LOS_truncation_factor=50):
        """
        This implements a tidal truncation at r_N, where N is some overdensity with respect to the critical density of
        the Universe at z
        :param lens_cosmo: an instance of LensCosmo
        :param LOS_truncation_factor: the multiple of the overdensity threshold at which to truncte the halo, e.g. N=200
        would truncate at r200
        """
        self._lens_cosmo = lens_cosmo
        self._N = LOS_truncation_factor

    def truncation_radius_halo(self, halo):
        """
        Thiis method computes the truncation radius using the class attributes of an instance of Halo
        :param halo: an instance of halo
        :return: the truncation radius in physical kpc
        """
        return self.truncation_radius(halo.mass, halo.z)

    def truncation_radius(self, halo_mass, z):
        """
        Computes the radius r_N of an NFW halo
        :param halo_mass: halo mass (m200 with respect to critical density at z)
        :param z: redshift
        :param lens_cosmo: an instance of LensCosmo
        :return: the truncation radius
        """
        h = self._lens_cosmo.cosmo.h
        rN_physical_mpc = self._lens_cosmo.rN_M(halo_mass * h, z, self._N)
        return rN_physical_mpc*1000

class TruncationRoche(object):

    def __init__(self, lens_cosmo=None, RocheNorm=1.4, m_power=1./3, RocheNu=2./3):
        """
        This implements a tidal truncation for subhalos of the form

        r_t = norm * (m / 10^7) ^ m_power * (r3d/50 kpc)^r3d_power [kpc]

        The default values were calibrated for a subhalo inside an isothermal host potential
        https://ui.adsabs.harvard.edu/abs/2016PhRvD..94d3505C/abstract

        :param norm: the overall scale
        :param m_power: exponent for the dependence on halo mass
        :param r3d_power: exponent for the dependence on 3D position inside host
        """
        self._norm = RocheNorm
        self._m_power = m_power
        self._r3d_power = RocheNu

    def truncation_radius_halo(self, halo):

        """
        Thiis method computess the truncation radius using the class attributes of an instance of Halo
        :param halo: an instance of halo
        :return: the truncation radius in physical kpc
        """
        return self.truncation_radius(halo.mass, halo.r3d)

    def truncation_radius(self, subhalo_mass, subhalo_r3d):

        """
        :param M: m200
        :param r3d: 3d radial position in the halo (physical kpc)
        :return: the truncation radius in physical kpc
        """
        m_units_7 = subhalo_mass / 10 ** 7
        radius_units_50 = subhalo_r3d / 50
        rtrunc_kpc = self._norm * m_units_7 ** self._m_power * radius_units_50 ** self._r3d_power
        return np.round(rtrunc_kpc, 3)

class AdiabaticTidesTruncation(object):
    """
    An example of the type of class we want to create and implement in pyHalo
    """

    def __init__(self, lens_cosmo, log_m_host, z_host):
        """

        :param lens_cosmo:
        :param log_m_host:
        :param z_host:
        """

        self._lens_cosmo = lens_cosmo
        cmodel = ConcentrationDiemerJoyce(self._lens_cosmo.cosmo.astropy, scatter=False)
        c_host = cmodel.nfw_concentration(10**log_m_host, z_host)
        self._c_host = c_host
        self._host_dynamical_time = self._lens_cosmo.halo_dynamical_time(10**log_m_host, z_host, c_host)
        self._tau_mf_interpolation = tau_mf_interpolation()
        self._min_c = 1.0
        self._max_c = 300.0

    def _mass_loss(self, a, b, s, c_halo):
        """
        Returns an approximation of the mass loss
        :param a:
        :param b:
        :param s:
        :param c_halo:
        :return:
        """
        log10c = np.log10(c_halo)
        arg = (log10c - a) / b / 2
        return s/2 * np.sqrt(c_halo/self._c_host) * (1 + np.tanh(arg))

    def mass_loss(self, c_halo, rperi_units_r200):
        """

        :param c_halo:
        :param rperi_units_r200:
        :return:
        """
        log10rp = np.log10(rperi_units_r200)
        log10rp = max(-2.0, log10rp)
        log10rp = min(0.0, log10rp)
        x1 = 1.51
        x2 = 2.01
        x3 = 3.0
        c1 = 0.16
        c2 = 0.27
        c3 = 0.38
        a = 0.65 * ( (log10rp / x1) ** c1 + (log10rp / x2) ** c2 + (log10rp / x3) ** c3)
        b = -0.005 - 0.075 * log10rp
        u1 = 0.1 - log10rp
        u2 = 0.6 - log10rp
        s = 0.048 * (u1 ** -0.3 + u2 **-1.5)
        return self._mass_loss(a, b, s, c_halo)

    def truncation_radius_halo(self, halo):
        """
        This function solves for the truncation radius divided by the halo scale radius (tau) given an instance of
        Halo (see pyhalo.HaloModels.TNFW) and other arguments for the interpolation function

        :param halo: an instance of the Halo class (should be TNFW)

        :return tau: the truncation radius divided by the halo's scale radius
        """

        # compute the halo parameeters
        c = halo.c
        # now make sure that the points are inside the region where we computed the interpolation
        r_pericenter_over_r200 = np.absolute(halo.rperi_units_r200)

        # evaluate the mass loss
        log10mass_loss_fraction_asymptotic = self.mass_loss(c, r_pericenter_over_r200)

        time_since_infall = halo.time_since_infall
        n_orbits = time_since_infall / self._host_dynamical_time
        mass_loss_fraction = self._temporal_mass_loss(10 ** log10mass_loss_fraction_asymptotic, n_orbits)
        log10mass_loss_fraction = np.log10(mass_loss_fraction)

        # solve for tau
        log10c = np.log10(c)
        point = (log10c, log10mass_loss_fraction)
        point = self._make_params_in_bounds_tau_evaluate(point)
        log10tau = float(self._tau_mf_interpolation(point))
        tau = 10 ** log10tau
        _, rs, _ = self._lens_cosmo.NFW_params_physical(halo.mass, halo.c, halo.z_eval)
        return tau * rs

    @staticmethod
    def _temporal_mass_loss(mass_loss_asymptotic, n_orbits):
        """
        This routine interpolates between zero mass loss at infall and the asymptotic value
        predicted by the adiabatic tides model
        """
        mass_loss = mass_loss_asymptotic * (1 / mass_loss_asymptotic) ** (1. / (1. + 0.6 * n_orbits))
        return mass_loss

    def _make_params_in_bounds_tau_evaluate(self, point):
        """
        This routine makes sure the arguments for the initerpolation are inside the domain of the function.
        """
        (log10c, log10mass_loss_fraction) = point
        log10c = max(self._min_c, log10c)
        log10c = min(self._max_c, log10c)
        log10mass_loss_fraction = max(-1.5, log10mass_loss_fraction)
        log10mass_loss_fraction = min(-0.01, log10mass_loss_fraction)
        return (log10c, log10mass_loss_fraction)
