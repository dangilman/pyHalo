import numpy as np


class TruncationRN(object):

    def __init__(self, LOS_truncation_factor=50):
        """
        This implements a tidal truncation at r_N, where N is some overdensity with respect to the critical density of
        the Universe at z
        """
        self._N = LOS_truncation_factor

    def truncation_radius_halo(self, halo):
        """
        Thiis method computess the truncation radius using the class attributes of an instance of Halo
        :param halo: an instance of halo
        :return: the truncation radius
        """
        return self.truncation_radius(halo.mass, halo.z, halo.r3d, halo._lens_cosmo)

    def truncation_radius(self, halo_mass, z, lens_cosmo):
        """
        Computes the radius r_N of an NFW halo
        :param halo_mass: halo mass (m200 with respect to critical density at z)
        :param z: redshift
        :param lens_cosmo: an instance of LensCosmo
        :return: the truncation radius
        """
        a_z = lens_cosmo.cosmo.scale_factor(z)
        h = lens_cosmo.cosmo.h
        r50_physical_Mpc = self.lens_cosmo.rN_M_nfw_comoving(M * h, N, z) * a_z / h
        rN_physical_kpc = r50_physical_Mpc * 1000
        return rN_physical_kpc

class TruncationRoche(object):

    def __init__(self, RocheNorm=1.4, m_power=1./3, RocheNu=2./3):
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
        :return: the truncation radius
        """
        return self.truncation_radius(halo.mass, halo.r3d)

    def truncation_radius(self, subhalo_mass, subhalo_r3d):

        """
        :param M: m200
        :param r3d: 3d radial position in the halo (kpc)
        :return: truncation radius in Kpc (physical)
        """
        m_units_7 = subhalo_mass / 10 ** 7
        radius_units_50 = subhalo_r3d / 50
        rtrunc_kpc = self._norm * m_units_7 ** self._m_power * radius_units_50 ** self._r3d_power
        return np.round(rtrunc_kpc, 3)
