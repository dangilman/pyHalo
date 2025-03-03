import numpy as np
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from pyHalo.Halos.tnfw_halo_util import tau_mf_interpolation
from colossus.lss import peaks
from colossus.halo import splashback
from pyHalo.Halos.galacticus_truncation.interp_mass_loss import InterpGalacticus, InterpGalacticusKeeley24
from pyHalo.Halos.galacticus_truncation.transfer_function_density_profile import compute_r_te_and_f_t


class TruncationSplashBack(object):
    name = "TruncationSplashBack"
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
        Computes the truncation radius as the splashback radius an NFW halo
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
    name = "TruncationRN"
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
        return self.truncation_radius(halo.mass, halo.z_eval)

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
    name = "Roche"
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

class TruncateMeanDensity(object):
    name = "MeanDensity"
    def __init__(self, lens_cosmo, median_rt_over_rs=2.0, c_power=3.0, intrinsic_scatter=0.0):
        """
        Implements a truncation model where the truncation radius depends on the infall concentration and orbital pericenter
        r_t / r_s = norm * (c / c_med)^c_power * (r_peri / (0.5 * r_200))^rperi_power
        :param lens_cosmo: an instance of LensCosmo
        :param median_rt_over_rs: the value of rt / rs for a halo that falls into the host with the median concetration
        for halos of its mass with an orbital pericenter of half the host's virial radius
        :param c_power: scales the dependence on infall concentration
        :param intrinsic_scatter: adds Gaussian scatter to rt_over_rs in unis of rt_over_rs, i.e.
        if intrinsic_scatter = s the scatter is given by Gaussian(mean = 0.0, sigma= s * rt_over_rs)
        """

        self._norm = median_rt_over_rs
        self._cpower = c_power
        self.lens_cosmo = lens_cosmo
        self._scatter = intrinsic_scatter
        self._concentration_cdm = ConcentrationDiemerJoyce(lens_cosmo.cosmo,
                                                           scatter=False)

    def truncation_radius_halo(self, halo):

        """
        Thiis method computess the truncation radius using the class attributes of an instance of Halo
        :param halo: an instance of halo
        :return: the truncation radius in physical kpc
        """
        c_median = self._concentration_cdm.nfw_concentration(halo.mass, halo.z_eval)
        c_actual = halo.c
        return self.truncation_radius(halo.mass, halo.z_eval, c_median, c_actual, self._scatter)

    def truncation_radius(self, halo_mass, z_eval, c_median, c_actual, intrinsic_scatter):
        """
        Computes the truncation radius from halo mass, halo redshift, median concentration at infall for halos of the
        given mass, and the actual concentration of the halo at infall, plus Gaussian scatter
        :param halo_mass: halo mass at infall
        :param z_eval: redshift of infall for subhalos, or redshift of field halo
        :param c_median: median concentration of halos of mass halo_mass at z = z_infall in CDM
        :param c_actual: actual concentration of the halo at infall
        :param intrinsic_scatter: Gaussian scatter
        :return:
        """
        _rt_over_rs = self._norm * (c_actual / c_median) ** self._cpower
        if intrinsic_scatter > 0:
            rt_over_rs = abs(np.random.normal(_rt_over_rs, intrinsic_scatter * _rt_over_rs))
        else:
            rt_over_rs = _rt_over_rs
        _, rs, _ = self.lens_cosmo.NFW_params_physical(halo_mass, c_actual, z_eval)
        return rs * rt_over_rs

class Multiple_RS(object):
    name = "Multiple_RS"
    """
    Sets the concentration to a constant multiple of the halo scale radius; note this will only work with a class
    that has defined NFW parameters
    """
    def __init__(self, lens_cosmo, tau):
        self.tau = tau
    def truncation_radius_halo(self, halo):
        nfw_params = halo.nfw_params
        rs = nfw_params[1]
        return self.tau * rs
    def truncation_radius(self, rs):
        return self.tau * rs

class ConstantTruncationArcsec(object):
    name = "Constant"
    """A constant truncation radius in arcsec"""
    def __init__(self, lens_cosmo, rt_arcsec):
        self.rt_arcsec = rt_arcsec
        pass
    def truncation_radius_halo(self, halo):
        return self.rt_arcsec
    def truncation_radius(self, *args, **kwargs):
        return self.rt_arcsec

class TruncationGalacticusKeeley24(object):
    name = "TruncationGalacticusKeeley24"
    def __init__(self, lens_cosmo, c_host):
        """

        :param lens_cosmo:
        """

        self._chost = c_host
        self._lens_cosmo = lens_cosmo
        self._mass_loss_interp = InterpGalacticusKeeley24()
        self._tau_mf_interpolation = tau_mf_interpolation()

    @staticmethod
    def _make_params_in_bounds_tau_evaluate(point):
        """
        This routine makes sure the arguments for the interpolation are inside the domain of the function.
        """
        min_c, max_c = np.log10(1.0), np.log10(10 ** 2.7)
        (log10c, log10mass_loss_fraction) = point
        log10c = max(min_c, log10c)
        log10c = min(max_c, log10c)
        log10mass_loss_fraction = max(-8.0, log10mass_loss_fraction)
        log10mass_loss_fraction = min(-0.001, log10mass_loss_fraction)
        return (log10c, log10mass_loss_fraction)

    def calculate_mbound(self, halo):
        """
        compute the bound mass
        :param halo:
        :return:
        """
        halo_mass = halo.mass
        infall_concentration = halo.c
        time_since_infall = halo.time_since_infall
        log10c = np.log10(infall_concentration)
        log10mbound_over_minfall = self._mass_loss_interp(log10c, time_since_infall, self._chost)
        m_bound = halo_mass * 10 ** log10mbound_over_minfall
        return m_bound

    def truncation_radius_halo(self, halo):

        """
        Thiis method computess the truncation radius using the class attributes of an instance of Halo
        :param halo: an instance of halo
        :return: the truncation radius in physical kpc
        """

        return self.truncation_radius(halo.mass, halo.c,
                                      halo.time_since_infall, self._chost, halo.z_eval)

    def truncation_radius(self, halo_mass, infall_concentration,
                          time_since_infall, chost, z_eval):
        """

        :param halo_mass:
        :param infall_concentration:
        :param time_since_infall:
        :param chost:
        :param z_eval:
        :return:
        """
        log10c = np.log10(infall_concentration)
        log10mbound_over_minfall = self._mass_loss_interp(log10c, time_since_infall, chost)
        point = self._make_params_in_bounds_tau_evaluate((log10c, log10mbound_over_minfall))
        log10tau = float(self._tau_mf_interpolation(point))
        tau = 10 ** log10tau
        _, rs, _ = self._lens_cosmo.NFW_params_physical(halo_mass,
                                                        infall_concentration,
                                                        z_eval)
        return tau * rs

class TruncationGalacticus(object):
    name = "TruncationGalacticus"
    def __init__(self, lens_cosmo, c_host):
        """

        :param lens_cosmo: an instance of LensCosmo
        :param c_host: host halo concentration at z=0.5
        """
        self._chost = c_host
        self._lens_cosmo = lens_cosmo
        self._mass_loss_interp = InterpGalacticus()
        self._tau_mf_interpolation = tau_mf_interpolation()

    @staticmethod
    def _make_params_in_bounds_tau_evaluate(point):
        """
        This routine makes sure the arguments for the interpolation are inside the domain of the function.
        """
        min_c, max_c = np.log10(1.0), np.log10(10 ** 2.7)
        (log10c, log10mass_loss_fraction) = point
        log10c = max(min_c, log10c)
        log10c = min(max_c, log10c)
        log10mass_loss_fraction = max(-8.0, log10mass_loss_fraction)
        log10mass_loss_fraction = min(-0.001, log10mass_loss_fraction)
        return (log10c, log10mass_loss_fraction)

    def calculate_mbound(self, halo):
        """
        compute the bound mass
        :param halo:
        :return:
        """
        halo_mass = halo.mass
        infall_concentration = halo.c
        time_since_infall = halo.time_since_infall
        log10c = np.log10(infall_concentration)
        log10mbound_over_minfall = self._mass_loss_interp(log10c, time_since_infall, self._chost)
        m_bound = halo_mass * 10 ** log10mbound_over_minfall
        return m_bound

    def truncation_radius_halo(self, halo, psuedo_nfw=False):
        """
        Thiis method computess the truncation radius using the class attributes of an instance of Halo
        :param halo: an instance of halo
        :return: the truncation radius in physical kpc
        """

        halo_mass = halo.mass
        infall_concentration = halo.c
        time_since_infall = halo.time_since_infall
        log10c = np.log10(infall_concentration)
        log10mbound_over_minfall = self._mass_loss_interp(log10c, time_since_infall, self._chost)
        m_bound = halo_mass * 10 ** log10mbound_over_minfall
        _, rs, r200 = self._lens_cosmo.NFW_params_physical(halo_mass,
                                                           infall_concentration,
                                                           halo.z,
                                                           psuedo_nfw)
        r_te, f_t = compute_r_te_and_f_t(m_bound, halo_mass, r200, infall_concentration)
        halo.rescale_normalization(f_t)
        return r_te

    def truncation_radius_from_bound_mass(self, halo, m_bound, psuedo_nfw=False):
        """
        Thiis method computes the truncation radius given the bound mass of a halo
        :param halo: an instance of a Halo class
        :param m_bound: the bound mass in solar masses
        :param psuedo_nfw: see documentation in LensCosmo / nfw_parameter classes
        :return: the truncation radius in physical kpc
        """
        halo_mass = halo.mass
        infall_concentration = halo.c
        _, rs, r200 = self._lens_cosmo.NFW_params_physical(halo_mass,
                                                           infall_concentration,
                                                           halo.z_eval,
                                                           psuedo_nfw)
        r_te, f_t = compute_r_te_and_f_t(m_bound, halo_mass, r200, infall_concentration)
        halo.rescale_normalization(f_t)
        return r_te

    def truncation_radius(self, halo_mass, infall_concentration,
                          time_since_infall, z_eval, psuedo_nfw=False):
        """

        :param halo_mass:
        :param infall_concentration:
        :param time_since_infall:
        :param chost:
        :param z_eval:
        :return:
        """
        log10c = np.log10(infall_concentration)
        log10mbound_over_minfall = self._mass_loss_interp(log10c, time_since_infall, self._chost)
        m_bound = halo_mass * 10 ** log10mbound_over_minfall
        _, rs, r200 = self._lens_cosmo.NFW_params_physical(halo_mass,
                                                        infall_concentration,
                                                        z_eval,
                                                        psuedo_nfw)
        r_te, _ = compute_r_te_and_f_t(m_bound, halo_mass, r200, infall_concentration)
        return r_te



