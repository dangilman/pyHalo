from colossus.lss.mass_function import *
from pyHalo.Cosmology.geometry import *
from scipy.interpolate import interp1d
from pyHalo.defaults import *
from colossus.lss.bias import twoHaloTerm

class LensingMassFunction2(object):

    def __init__(self, cosmology, mlow, mhigh, zlens, zsource, cone_opening_angle,
                 mass_function_model=None, use_lookup_table=True, two_halo_term = True,
                 geometry_type=None):

        self._cosmo = cosmology

        if mass_function_model is None:
            mass_function_model = realization_default.default_mass_function

        if geometry_type is None:
            geometry_type = lenscone_default.default_geometry

        self.geometry = Geometry(cosmology, zlens, zsource, cone_opening_angle, geometry_type)
        self._mass_function_model = mass_function_model

        self._mlow, self._mhigh = mlow, mhigh
        self._two_halo_term = two_halo_term

        self._delta_z = 0.05

        self._z_range, self._logM_decades, self._interpolated_norm, \
        self._interpolated_slopes = self._build(self._delta_z, zsource)

    def norm_at_z_density(self, z, log_mass):

        i = self._interp_list_idx(log_mass)
        norm = self._interpolated_norm[i](z)
        return norm

    def plaw_index_z(self, z, log_mass):

        i = self._interp_list_idx(log_mass)
        norm = self._interpolated_slopes[i](z)
        return norm

    def _interp_list_idx(self, log_mass):

        if log_mass < self._logM_decades[0][0]:
            raise Exception('log_mass value '+str(log_mass)+' below minimum halo mass '
                            + str(self._logM_decades[0][0]))

        for i, dec in self._logM_decades:
            if log_mass < dec[1]:
                return i
        else:
            raise Exception('log_mass value ' + str(log_mass) + ' not in mass bins:'
                            + str(self._logM_decades))

    def _build(self, delta_z, zsource):

        start = lenscone_default.default_zstart
        # delta_z does not need to be equal to default z step
        z_range = np.arange(start, zsource + delta_z, delta_z)
        logM_decades = [[5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
        interpolated_norm = []
        interpolated_slopes = []

        for decade in logM_decades:
            log_norms, slopes = self._fit_z(z_range, decade[0], decade[1])
            interpolated_norm.append(interp1d(z_range, log_norms))
            interpolated_slopes.append(interp1d(z_range, slopes))

        return z_range, logM_decades, interpolated_norm, interpolated_slopes

    def norm_at_z(self, z, delta_z, log_mass):

        norm_dV = self.norm_at_z_density(z, log_mass)

        dV = self.geometry.volume_element_comoving(z, delta_z)

        return norm_dV * dV

    def norm_at_z_biased(self, z, delta_z, M_halo, log_mass, rmin = 0.5, rmax = 10):

        if self._two_halo_term:

            # factor of 2 for symmetry
            boost = 1+2*self.integrate_two_halo(M_halo, z, rmin=rmin, rmax=rmax) / (rmax - rmin)

            return boost * self.norm_at_z(z, delta_z, log_mass)
        else:
            return self.norm_at_z(z, delta_z, log_mass)

    def integrate_two_halo(self, m200, z, rmin = 0.5, rmax = 10):

        def _integrand(x):
            return self.twohaloterm(x, m200, z)

        boost = quad(_integrand, rmin, rmax)[0]

        return boost

    def twohaloterm(self, r, M, z, mdef='200c'):

        h = self._cosmo.h
        M_h = M * h
        r_h = r * h

        rho_2h = twoHaloTerm(r_h, M_h, z, mdef=mdef) * self._cosmo._colossus_cosmo.rho_m(z) ** -1

        return rho_2h * h ** -2

    def _fit_z(self, zvalues, logm1, logm2):

        norms, slopes = [], []
        for zi in zvalues:

            M = np.logspace(logm1, logm2, 10)
            dN_dMdV = self.dN_dMdV_comoving(M, zi)
            [plaw_idx, log_norm] = np.polyfit(np.log10(M), np.log10(dN_dMdV), 1)

            norms.append(log_norm)
            slopes.append(plaw_idx)

        return np.array(norms), np.array(slopes)

    def dN_dMdV_comoving(self, M, z):

        """
        :param M: m (in physical units, no little h)
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (physical)
        [N * M_sun ^ -1 * Mpc ^ -3]
        """

        h = self._cosmo.h

        M_h = M*h

        dndM_h = h ** 3 * massFunction(M_h, z, q_out='dndlnM') * M_h ** -1
        return h * dndM_h

    def dN_comoving_deltaFunc(self, M, z, delta_z, component_fraction):

        """

        :param z: redshift
        :param component_fraction: density parameter; fraction of the matter density (not fraction of critical density!)
        :return: the number of objects of mass M * Mpc^-3
        """

        dN_dV = self._cosmo.rho_matter_crit(z) * component_fraction * M ** -1

        return dN_dV * self.geometry.volume_element_comoving(z, delta_z)

    def _fit_norm_index(self, M, dNdM, order=1):
        """
        Fits a line to a log(M) log(dNdM) relation
        :param M: masses
        :param dNdM: mass function
        :param order: order of the polynomial
        :return: normalization units [dNdM * dM ^ -1], power law exponent
        """

        if np.all(dNdM==0):
            return 0, 2

        coeffs = np.polyfit(np.log10(M), np.log10(dNdM), order)
        plaw_index = coeffs[0]
        norm = 10 ** coeffs[1]

        return norm,plaw_index

    def integrate_mass_function(self, z, delta_z, mlow, mhigh, log_m_break, break_index, break_scale, n=1,
                                norm_scale = 1):

        m_break = 10**log_m_break

        def _integrand(m):
            logm = np.log10(m)
            norm = self.norm_at_z(z, delta_z, logm)
            plaw_index = self.plaw_index_z(z, logm)
            norm_scaled = norm * norm_scale
            return norm_scaled * m ** (n + plaw_index) * \
                   (1 + break_scale * m_break / m) ** break_index

        moment = quad(_integrand, mlow, mhigh)[0]
        return moment

    def integrate_power_law(self, norm, m_low, m_high, log_m_break, n, plaw_index, break_index=0, break_scale=1):

        def _integrand(m, m_break, plaw_index, n):

            return norm * m ** (n + plaw_index) * (1 + break_scale * m_break / m) ** break_index

        moment = quad(_integrand, m_low, m_high, args=(10**log_m_break, plaw_index, n))[0]

        return moment
