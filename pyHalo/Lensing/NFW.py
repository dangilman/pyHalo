import numpy as np
from colossus.halo.profile_nfw import NFWProfile

class NFWLensing(object):

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = lens_cosmo

    def params(self, x, y, mass, concentration, redshift):

        theta_Rs, Rs_angle = self.nfw_physical2angle(mass, concentration, redshift)

        kwargs = {'theta_Rs':theta_Rs, 'Rs': Rs_angle,
                  'center_x':x, 'center_y':y}

        return kwargs

    def nfw_params_physical_colossus(self, m, c, z, mdef='200c'):

        h = self.lens_cosmo.cosmo.h

        m_h = m * h

        nfw = NFWProfile(m_h, c, z, mdef)
        # retuns the physical density and scale radius in kpc
        # units are: h^2 M_sun / kpc ^ 3, kpc / h
        rhos_h2, rs_hinv = nfw.fundamentalParameters(m_h, c, z, mdef=mdef)

        rhos = rhos_h2 * h ** -2

        rs = rs_hinv * h

        return rhos, rs, rs*c

    def nfw_physical2angle(self, m, c, z):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """

        if z < 1e-4:
            z = 1e-4

        theta_rs, rs_angle = self.lens_cosmo.nfw_physical2angle(m, c, z)
        return theta_rs, rs_angle

    def M_physical(self, m, c, z):
        """

        :param m200: m200
        :return: physical mass corresponding to m200
        """

        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)



