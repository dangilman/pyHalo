from colossus.lss.mass_function import *
from pyHalo.Cosmology.geometry import *
from scipy.interpolate import interp1d
from pyHalo.defaults import *
from colossus.lss.bias import twoHaloTerm, haloBias

class LensingMassFunction(object):

    def __init__(self,cosmology,mlow,mhigh,zlens,zsource,cone_opening_angle,
                 delta_theta_lens=None,model='reed07',model_kwargs={}):

        if delta_theta_lens is None:
            delta_theta_lens = cone_opening_angle

        self._cosmo = cosmology
        self.geometry = Geometry(cosmology, zlens, zsource, delta_theta_lens, cone_opening_angle)
        self._model = model
        self._model_kwargs = model_kwargs
        self._mlow, self._mhigh = mlow, mhigh
        self._M = np.logspace(np.log10(mlow), np.log10(mhigh), 100)

        # densities
        norm_z_dV, plaw_index_z, z_range, delta_z = self._build(mlow, mhigh, zsource, zlens)

        self._delta_z = delta_z
        self._norm_dV = interp1d(z_range,norm_z_dV)
        self._plaw_index_z = interp1d(z_range,plaw_index_z)
        self._z_range = z_range

    def total_los(self):

        n = 0
        for z in self._z_range:

            norm = self.norm_at_z(z,self._delta_z)
            plaw_idx = self.plaw_index_z(z)
            dNdM = norm*self._M ** plaw_idx
            n+=self._mass_function_moment(self._M, dNdM, 0, self._mlow, self._mhigh)

        return n

    def norm_at_z_density(self, z, units='comoving'):

        norm = self._norm_dV(z)

        if units == 'comoving':
            a_scale = 1
        elif units == 'physical':
            a_scale = self._cosmo.scale_factor(z) ** -3

        return a_scale * norm

    def norm_at_z(self, z, delta_z):

        norm_dV = self.norm_at_z_density(z)

        dV = self.geometry.volume_element_comoving(z, self.geometry._zlens, delta_z)

        return norm_dV * dV

    def norm_at_z_biased(self, z, M_halo, delta_z):

        delta_R = self.geometry.delta_R_fromz(z)

        norm_unbiased = self.norm_at_z(z, delta_z)

        if delta_R >= 100:
            return norm_unbiased
        elif z == self.geometry._zlens:
            return norm_unbiased
        else:

            delta_R = max(delta_R, 1e-4)
            boost = 1 + self.twohaloterm(delta_R,M_halo,z)
            return norm_unbiased * boost

    def plaw_index_z(self, z):

        return self._plaw_index_z(z)

    def twohaloterm(self, r, M, z, mdef='200c'):

        h = self._cosmo.h
        M_h = M * h
        r_h = r * h

        rho_2h = twoHaloTerm(r_h, M_h, z, mdef=mdef) * self._cosmo._colossus_cosmo.rho_m(z) ** -1

        return rho_2h * h ** -2

    def dN_dM(self, M, zstart, zend, z_lens):

        """

        :param M: Mass in solar masses
        :param zstart: start redshift
        :param zend: end redshift
        :param cone_opening_angle:
        :param z_lens:
        :param delta_theta_lens: reduced lens deflection angle
        :return: the mass function in units [N * M_sun ^ -1]
        """
        if zstart == 0:
            zstart = 1e-4

        delta_z = zend - zstart
        assert delta_z > 0

        # get the comoving volume element at redshift z
        volume_element_comoving = self.geometry.volume_element_comoving(zstart, z_lens, delta_z)

        return self.dN_dMdV_comoving(M, zstart) * volume_element_comoving

    def dN_dMdV_comoving(self, M, z):

        """
        :param M: m (in physical units, no little h)
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (comoving)
        [N * M_sun ^ -1 * Mpc ^ -3]
        """

        a_cube = self._cosmo.scale_factor(z)**3

        return a_cube*self._dN_dMdV_physical(M, z)

    def _build(self, mlow, mhigh, zsource, zlens):

        nsteps = int((zsource - default_zstart) * self.geometry._min_delta_z ** -1)
        z_range = np.linspace(default_zstart,zsource,nsteps)
        # omit the source redshift for numerical reasons

        delta_z = z_range[1]

        n, norm, index = [], [], []
        for i, zi in enumerate(z_range):
            zstart = zi
            zend = zi + delta_z
            ni, normi, indexi = self._mass_function_params(mlow, mhigh, zstart, zend, zlens)
            n.append(ni)
            norm.append(normi)
            index.append(indexi)

        return np.array(norm), np.array(index), z_range, delta_z


    def _mass_function_params(self, mlow, mhigh, zstart, zend, z_lens):

        """
        :param mlow: min mass in solar masses
        :param mhigh: maximum mass in solar masses
        :param zstart:
        :param zend:
        :param cone_opening_angle:
        :param z_lens:
        :param delta_theta_lens:
        :return: number of halos between mlow and mhigh, normalization and power law index of the mass function
        assumed parameterization is simple:

        dN / dM = norm * (M / M_sun) ^ plaw_index

        units of norm are therefore M_sun ^ -1
        """
        log_mlow = np.log10(mlow)
        log_mhigh = np.log10(mhigh)

        #dN_dM = self.dN_dM(M, zstart, zend, z_lens)
        dN_dMdV = self.dN_dMdV_comoving(self._M, zstart)

        #N_objects, norm, plaw_index = self._mass_function_moment(M, dN_dM, 0, mlow, mhigh)
        # returns x / mpc ^ 3
        N_objects_dV, norm_dV, plaw_index_dV = self._mass_function_moment(self._M, dN_dMdV, 0, mlow, mhigh)

        return N_objects_dV, norm_dV, plaw_index_dV

    def _dN_dMdV_physical(self, M, z):

        """
        :param M: m (in physical units, no little h)
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (physical)
        [N * M_sun ^ -1 * Mpc ^ -3]
        """

        h = self._cosmo.h

        M_h = M*h

        return h ** 3 * massFunction(M_h, z, q_out='dndlnM', model=self._model, **self._model_kwargs) * M_h ** -1

    def _dN_dMdV_comoving_deltaFunc(self, M, z, component_fraction):

        """

        :param z: redshift
        :param component_fraction: density parameter; fraction of the matter density (not fraction of critical density!)
        :return: the number of objects of mass M * Mpc^-3
        """
        return self._cosmo.rho_matter_crit(z) * component_fraction * M ** -1

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

    def _mass_function_moment(self, M, dNdM, n, m_low, m_high, order=1):

        """
        :param normalization: dimensions M_sun^-1 Mpc^-3 or M_sun
        :param plaw_index: power law index
        :param N: for the Nth moment
        :return: Nth moment of the mass funciton
        """

        norm,plaw_index = self._fit_norm_index(M, dNdM, order=order)

        if plaw_index == 2 and n==1:
            moment = norm * np.log(m_high * m_low ** -1)

        else:
            newindex = 1 + n + plaw_index

            moment = norm * newindex ** -1 * (m_high ** newindex - m_low ** newindex)

        return moment,norm,plaw_index

    def _unit_to_unit_littleh(self, unit):

        return unit * self._cosmo.h

    def _unit_littleh_to_unit(self, unit_h):

        return unit_h * self._cosmo.h ** -1

    def _dN_dMdV_example(self, M, z):

        """
        :param M: M (physical m200, in solar masses, no little h)
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (physical? comoving?)
        [N * M_sun ^ -1 * Mpc ^ -3]

        The intention is that the quantity returned times a small delta_M is the number of halos within a
        cubic megaparsec (physical) in the mass range [M, M + delta_M] in physical solar masses
        """

        # h = 0.679
        h = self._cosmo.h

        # convert M to M/h units
        m_h = M * h

        dn_dlogm_h = massFunction(m_h, z, q_out='dndlnM', model='reed07')

        # convert (Mpc / h)^-3 to Mpc^-3
        dn_dlogm_h *= h**3

        return dn_dlogm_h / m_h

