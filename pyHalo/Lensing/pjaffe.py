import numpy as np

class PJaffeLensing(object):

    hybrid = False

    lenstronomy_ID = 'PJAFFE'

    def __init__(self, lens_cosmo):

        self._lens_cosmo = lens_cosmo

    def params(self, x, y, mass, redshift, concentration, r_trunc_kpc):

        center_x = x
        center_y = y

        _, rs_kpc, r200_kpc = self._lens_cosmo.NFW_params_physical(mass, concentration, redshift)

        r_a_kpc = 0.005 * rs_kpc
        Sigma0 = mass/rs_kpc/(2*np.pi*r_a_kpc) # units M_sun / kpc^2

        sigma_crit = self._lens_cosmo.get_sigma_crit_lensing_kpc(redshift, self._lens_cosmo.z_source)
        sigma_0 = Sigma0/sigma_crit_kpc

        kpc_to_arcsec = 1/self._lens_cosmo.cosmo.kpc_proper_per_asec(redshift)
        r_trunc_arcsec = r_trunc_kpc * kpc_to_arcsec
        r_a_arcsec = r_a_kpc * kpc_to_arcsec

        return {'center_x':center_x, 'center_y': center_y, 'Ra': r_a_arcsec,
                'Rs': r_trunc_arcsec, 'sigma0': sigma_0}, None
