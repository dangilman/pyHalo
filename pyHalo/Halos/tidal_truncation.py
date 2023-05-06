import numpy as np
import inspect
import pickle
import inspect
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from scipy.interpolate import RegularGridInterpolator
from pyHalo.Halos.util import tau_mf_interpolation
_path = inspect.getfile(inspect.currentframe())[0:-29]+'/adiabatic_tides_data/'


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
        # concentration doesn't matter here
        _, _, rN_physical_mpc = self._lens_cosmo.nfwParam_physical(halo_mass, 16.0, z)
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

    def __init__(self, lens_cosmo, m_host, z_host, log10_galaxy_rs, log10_galaxy_m):
        """

        :param lens_cosmo:
        :param m_host:
        :param z_host:
        :param log10_galaxy_rs:
        :param log10_galaxy_m:
        """

        m_host_list = np.array([13.0])
        z_host_list = np.array([0.5])
        fnames = ['13.0_z0.5']

        min_max_c = [1.0, 10 ** 2.7]
        min_max_rperi = [10 ** -2.5, 1.0]

        fname_base = 'subhalo_mass_loss_interp_mhost'
        dmhost = abs(m_host_list - np.log10(m_host)) / 0.1
        d_zhost = abs(z_host_list - z_host) / 0.2
        penalty = dmhost + d_zhost
        idx_min = np.argsort(penalty)[0]
        fname = fname_base + fnames[idx_min]
        f = open(_path + fname, 'rb')
        self._mass_loss_interp = pickle.load(f)
        f.close()

        self._lens_cosmo = lens_cosmo
        cmodel = ConcentrationDiemerJoyce(self._lens_cosmo.cosmo.astropy, scatter=False)
        c_host = cmodel.nfw_concentration(m_host, z_host)
        self._host_dynamical_time = self._lens_cosmo.halo_dynamical_time(m_host, z_host, c_host)
        self._min_c = min_max_c[0]
        self._max_c = min_max_c[1]
        self._min_rperi = min_max_rperi[0]
        self._max_rperi = min_max_rperi[1]
        self._log10_galaxy_rs = log10_galaxy_rs
        self._log10_galaxy_m = log10_galaxy_m
        self._tau_mf_interpolation = tau_mf_interpolation()

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
        if r_pericenter_over_r200 > self._max_rperi:
            r_pericenter_over_r200 = self._max_rperi
        if r_pericenter_over_r200 < self._min_rperi:
            r_pericenter_over_r200 = self._min_rperi
        point = (np.log10(c), np.log10(r_pericenter_over_r200), self._log10_galaxy_rs, self._log10_galaxy_m)
        point = self._make_params_in_bounds(point)

        # evaluate the mass loss
        log10mass_loss_fraction_asymptotic = float(self._mass_loss_interp(point))
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
        if 10 ** log10c < self._min_c:
            log10c = np.log10(self._min_c)
        elif 10 ** log10c > self._max_c:
            log10c = np.log10(self._max_c)
        if log10mass_loss_fraction < -1.5:
            log10mass_loss_fraction = -1.5
        elif log10mass_loss_fraction > -0.01:
            log10mass_loss_fraction = -0.01
        return (log10c, log10mass_loss_fraction)

    def _make_params_in_bounds(self, point):
        """
        This routine makes sure the arguments for the initerpolation are inside the domain of the function.
        """
        (log10c, log10r_pericenter_over_r200, log10_galaxy_rs, log10_galaxy_m) = point
        if log10c < np.log10(self._min_c):
            log10c = np.log10(self._min_c)
        elif log10c > np.log10(self._max_c):
            log10c = np.log10(self._min_c)
        if log10r_pericenter_over_r200 < -2.5:
            log10r_pericenter_over_r200 = -2.5
        elif log10r_pericenter_over_r200 > 0.0:
            log10r_pericenter_over_r200 = 0.0
        if log10_galaxy_rs < -2.0:
            log10_galaxy_rs = -2.0
        elif log10_galaxy_rs > 0.3:
            log10_galaxy_rs = 0.3
        if log10_galaxy_m < -2.5:
            log10_galaxy_m = -2.5
        elif log10_galaxy_m > -0.25:
            log10_galaxy_m = -0.25
        new_point = (log10c, log10r_pericenter_over_r200, log10_galaxy_rs, log10_galaxy_m)
        return new_point
