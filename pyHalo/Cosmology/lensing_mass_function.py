from colossus.lss.mass_function import *
from pyHalo.Cosmology.geometry import *
from scipy.interpolate import interp1d
from pyHalo.defaults import *
from colossus.lss.bias import twoHaloTerm

class LensingMassFunction(object):

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

        self._norms_z_dV, self._plaw_indexes_z, self._log_mbin = [], [], []

        if use_lookup_table:

            if self._mass_function_model == 'sheth99':
                from pyHalo.Cosmology.lookup_tables import lookup_sheth99_simple as table
            else:
                raise ValueError('lookup table '+self._mass_function_model+' not found.')

            norm_z_dV = table.norm_z_dV
            plaw_index_z = table.plaw_index_z
            z_range = table.z_range
            delta_z = table.delta_z

        else:
            # list ordering is by mass, with sublists consisting of different redshifts
            norm_z_dV, plaw_index_z, z_range, delta_z = self._build(mlow, mhigh, zsource)

            self._norm_z_dV = norm_z_dV
            self._plaw_index_z = plaw_index_z
            self._z_range = z_range
            self._delta_z = delta_z

        self._norm_dV_interp = interp1d(z_range, norm_z_dV)

        self._plaw_interp = interp1d(z_range, plaw_index_z)

        self._delta_z = delta_z

        self._z_range = z_range

    def norm_at_z_density(self, z):

        norm = self._norm_dV_interp(z)

        return norm

    def plaw_index_z(self, z):

        idx = self._plaw_interp(z)

        return idx

    def norm_at_z(self, z, delta_z):

        norm_dV = self.norm_at_z_density(z)

        dV = self.geometry.volume_element_comoving(z, delta_z)

        return norm_dV * dV

    def norm_at_z_biased(self, z, delta_z, M_halo, rmin = 0.5, rmax = 10):

        if self._two_halo_term:

            # factor of 2 for symmetry
            boost = 1+2*self.integrate_two_halo(M_halo, z, rmin = rmin, rmax = rmax) / (rmax - rmin)

            return boost * self.norm_at_z(z, delta_z)
        else:
            return self.norm_at_z(z, delta_z)

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

    def _build(self, mlow, mhigh, zsource):

        z_range = np.arange(lenscone_default.default_zstart, zsource + 0.02, 0.02)

        #z_range = np.linspace(default_zstart, zsource - default_zstart, nsteps)

        M = np.logspace(8, np.log10(mhigh), 20)

        norm, index = [], []

        for zi in z_range:

            _, normi, indexi = self._mass_function_params(M, mlow, mhigh, zi)

            norm.append(normi)
            index.append(indexi)

        return norm, index, z_range, z_range[1] - z_range[0]

    def _mass_function_params(self, M, mlow, mhigh, zstart):

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

        dN_dMdV = self.dN_dMdV_comoving(M, zstart)

        N_objects_dV, norm_dV, plaw_index_dV = self._mass_function_moment(M, dN_dMdV, 0, mlow, mhigh)

        return N_objects_dV, norm_dV, plaw_index_dV

    def dN_dMdV_comoving(self, M, z):

        """
        :param M: m (in physical units, no little h)
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (physical)
        [N * M_sun ^ -1 * Mpc ^ -3]
        """

        h = self._cosmo.h

        M_h = M*h

        return h ** 3 * massFunction(M_h, z, q_out='dndlnM') * M_h ** -1

    def dN_comoving_deltaFunc(self, M, z, delta_z, component_fraction):

        """

        :param z: redshift
        :param component_fraction: density parameter; fraction of the matter density (not fraction of critical density!)
        :return: the number of objects of mass M * Mpc^-3
        """

        dN_dV = self._cosmo.rho_matter_crit(z) * component_fraction * M ** -1

        return dN_dV * self.geometry.volume_element_comoving(z, self.geometry._zlens, delta_z)

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

        norm = self.norm_at_z(z, delta_z)
        plaw_index = self.plaw_index_z(z)
        moment = self.integrate_power_law(norm_scale * norm, mlow, mhigh, log_m_break, n, plaw_index,
                                          break_index=break_index, break_scale=break_scale)

        return moment

    def integrate_power_law(self, norm, m_low, m_high, log_m_break, n, plaw_index, break_index=0, break_scale=1):

        def _integrand(m, m_break, plaw_index, n):

            return norm * m ** (n + plaw_index) * (1 + break_scale * m_break / m) ** break_index

        moment = quad(_integrand, m_low, m_high, args=(10**log_m_break, plaw_index, n))[0]

        return moment

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

            moment = self.integrate_power_law(norm, m_low, m_high, 0, n, plaw_index)

        return moment,norm,plaw_index

    def number_in_cylinder_arcsec(self, cylinder_diameter_arcsec, z_max, m_min=10**6.5, m_max=10**7.5,
                           z_min=0):

        dz = 0.01
        zsteps = np.arange(z_min+dz, z_max+dz, dz)
        nhalos = 0
        M = np.logspace(7, 10, 25)

        for zi in zsteps:

            dr = self.geometry._delta_R_comoving(zi, dz)
            cylinder_diameter_kpc = cylinder_diameter_arcsec*self.geometry._cosmo.kpc_per_asec(zi)
            radius = 0.5 * cylinder_diameter_kpc * 0.001
            dv_comoving = np.pi*radius**2 * dr
            nhalos += dv_comoving*self._mass_function_moment(M, self.dN_dMdV_comoving(M, zi), 0, m_min, m_max)[0]

        return nhalos

    def number_in_cylinder_kpc(self, cylinder_diameter_kpc, z_max, m_min=10**6.5, m_max=10**7.5,
                           z_min=0):

        dz = 0.01
        zsteps = np.arange(z_min+dz, z_max+dz, dz)
        nhalos = 0
        M = np.logspace(7, 10, 25)
        radius = 0.5 * cylinder_diameter_kpc * 0.001

        for zi in zsteps:

            dr = self.geometry._delta_R_comoving(zi, dz)
            dv_comoving = np.pi*radius**2 * dr
            nhalos += dv_comoving*self._mass_function_moment(M, self.dN_dMdV_comoving(M, zi), 0, m_min, m_max)[0]

        return nhalos

    def cylinder_volume(self, cylinder_diameter_kpc, z_max,
                           z_min=0):

        dz = 0.01
        zsteps = np.arange(z_min+dz, z_max+dz, dz)
        volume = 0
        for zi in zsteps:

            dr = self.geometry._delta_R_comoving(zi, dz)
            radius = 0.5 * cylinder_diameter_kpc * 0.001
            dv_comoving = np.pi*radius**2 * dr
            volume += dv_comoving

        return volume

def write_lookup_table():

    def write_to_file(fname, pname, values, mode):
        with open(fname, mode) as f:
            f.write(pname + ' = [')
            for j, val in enumerate(values):
                #if j % 13 == 0 and j >0:
                #    f.write('\ \n   ')
                f.write(str(val)+', ')
            f.write(']\n')

    from pyHalo.Cosmology.cosmology import Cosmology
    l = LensingMassFunction(Cosmology(), 10**5, 10**10, 0.2, 4, cone_opening_angle=6, use_lookup_table=False)

    fname = './lookup_tables/lookup_sheth99_simple.py'

    with open(fname, 'w') as f:
        f.write('import numpy as np\n')

    write_to_file(fname, 'norm_z_dV', l._norm_z_dV, 'a')
    write_to_file(fname, 'plaw_index_z', l._plaw_index_z, 'a')

    with open(fname, 'a') as f:
        f.write('z_range = np.array([')
        for zi in l._z_range:
            f.write(str(zi)+', ')
        f.write('])\n\n')

    with open(fname, 'a') as f:
        f.write('delta_z = '+str(l._delta_z)+'\n\n')

