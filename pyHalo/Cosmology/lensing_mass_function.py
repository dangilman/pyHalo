from colossus.lss.mass_function import *
from pyHalo.Cosmology.geometry import *
from scipy.interpolate import interp1d
from pyHalo.defaults import *
from colossus.lss.bias import twoHaloTerm
from scipy.integrate import simps

class LensingMassFunction(object):

    def __init__(self, cosmology, mlow, mhigh, zlens, zsource, cone_opening_angle,
                 m_pivot=10**8, mass_function_model=None, use_lookup_table=True, two_halo_term=True,
                 geometry_type=None):


        """
        This class handles computations of the halo mass function

        :param cosmology: An instance of Cosmology (see cosmology.py)
        :param mlow: low end of mass function
        :param mhigh: high end of mass function
        :param zlens: lens redshift
        :param zsource: source redshift
        :param cone_opening_angle: opening angle of lensing volume in arcseconds
        :param m_pivot: pivot mass of the mass function in M_sun
        :param mass_function_model (optional): the halo mass function model, default is Sheth-Tormen
        :param use_lookup_table: Whether to use a precomputed lookup table for the normalization and slope of the mass function
        :param two_halo_term: Whether to include the contribution from the two halo term of the main deflector
        :param geometry_type: Type of lensing geometry (DOUBLE_CONE, CYLINDER)
        """

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

        self.m_pivot = m_pivot

        if use_lookup_table and m_pivot == 10**8:

            if self._mass_function_model == 'sheth99':
                from pyHalo.Cosmology.lookup_tables import lookup_sheth99 as table
            else:
                raise Exception('lookup table '+self._mass_function_model+' not found.')

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

    def norm_at_z_density(self, z, plaw_index, m_pivot):

        """
        Returns the normalization of the mass function in units  Mpc^-3 / M_sun^-1
        :param z: redshift
        :param plaw_index: logarithmic slope of mass function
        :param m_pivot: pivot mass of mass function
        :return: normalization such that
        dN / dmdV = norm * m^plaw_index
        """

        norm = self._norm_dV_interp(z)

        assert m_pivot == self.m_pivot
        factor = 1/(m_pivot**plaw_index)

        return norm * factor

    def plaw_index_z(self, z):

        """

        :param z: redshift
        :return: the logarithmic slope of the halo mass function at redshift z
        """

        idx = self._plaw_interp(z)

        return idx

    def norm_at_z(self, z, plaw_index, delta_z, m_pivot):

        """

        :param z: redshift
        :param plaw_index: logarithmic slope of mass function
        :param delta_z: thickness of redshift slice
        :param m_pivot: pivot mass of mass function
        :return: the normalization of the halo mass function in units Mpc^-1
        such that

        dN / dm = norm * m^plaw_index

        """
        norm_dV = self.norm_at_z_density(z, plaw_index, m_pivot)

        dV = self.geometry.volume_element_comoving(z, delta_z)

        return norm_dV * dV

    def two_halo_boost(self, m200, z, rmin=0.5, rmax=10):

        """
        Computes the average contribution of the two halo term in a redshift slice adjacent
        the main deflector. Returns a rescaling factor applied to the mass function normalization

        :param m200: host halo mass
        :param z: redshift
        :param rmin: lower limit of the integral, something like the virial radius ~500 kpc
        :param rmax: Upper limit of the integral, this is computed based on redshift spacing during
        the rendering of halos
        :return: scaling factor applied to the normalization of the LOS mass function
        """
        def _integrand(x):
            return self.twohaloterm(x, m200, z)

        mean_boost = 2 * quad(_integrand, rmin, rmax)[0] / (rmax - rmin)
        # factor of two for symmetry in front/behind host halo

        return 1. + mean_boost

    def twohaloterm(self, r, M, z, mdef='200c'):

        """
        Computes the boost to the background density of the Universe
        from correlated structure around a host of mass M
        :param r:
        :param M:
        :param z:
        :param mdef:
        :return:
        """
        h = self._cosmo.h
        M_h = M * h
        r_h = r * h

        rho_2h = twoHaloTerm(r_h, M_h, z, mdef=mdef) / self._cosmo._colossus_cosmo.rho_m(z)

        return rho_2h

    def _build(self, mlow, mhigh, zsource):

        """
        This routine is used to interpolate the normalization and slope of the
         mass function as a function of redshift

        :param mlow: low end of mass function
        :param mhigh: high end of mass function
        :param zsource: source redshift
        :return: interpolation of halo mass function
        """

        z_range = np.arange(lenscone_default.default_zstart, zsource + 0.02, 0.02)

        #z_range = np.linspace(default_zstart, zsource - default_zstart, nsteps)

        M = np.logspace(np.log10(mlow), np.log10(mhigh), 20)

        norm, index, norm_scale = [], [], []

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

    def dN_dMdV_comoving(self, M, z, ps_args=None):

        """
        :param M: m (in physical units, no little h)
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (comoving)
        [N * M_sun ^ -1 * Mpc ^ -3]
        """

        h = self._cosmo.h

        # To M_sun / h units
        M_h = M*h

        if ps_args is not None:
            dndlogM = massFunction(M_h, z, q_out='dndlnM',
                                         ps_args=ps_args, model=self._mass_function_model)

        else:
            dndlogM = massFunction(M_h, z, q_out='dndlnM', model=self._mass_function_model)

        dndM_comoving_h = dndlogM / M_h
        # now we have an h^3 from the rho / M term in the mass function,
        # and a 1/(h^-1) = h from M_h, so four factors of h

        dndM_comoving = dndM_comoving_h * h ** 4

        return dndM_comoving

    def rho_dV(self, component_fraction):

        """

        :param z: redshift
        :param component_fraction: density parameter; fraction of the matter density (not fraction of critical density!)
        :return: the number of objects of mass M * Mpc^-3
        """

        #a_z = 1/(1+z)
        # in comoving units evaluate at z=0
        rho_dV = component_fraction * self._cosmo.rho_dark_matter_crit(0)

        return rho_dV

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

        coeffs = np.polyfit(np.log10(M/self.m_pivot), np.log10(dNdM), order)

        plaw_index = coeffs[0]
        norm = 10 ** coeffs[1]

        return norm, plaw_index

    def integrate_mass_function(self, z, plaw_index, delta_z, mlow, mhigh, log_m_break, break_index, break_scale, n=1,
                                norm_scale = 1):

        """
        Integrates the halo mass function between m_low and m_high

        :param norm: normalization prefactor
        :param m_low: low bound of integral
        :param m_high: high bound of integral
        :param log_m_break: characteristic break mass (in log) of the power law
        :param n: computes the nth moment
        :param plaw_index: logarithmic slope of power law
        :param break_index: see parameterization above
        :param break_scale: see parameterization above
        :return: the desired integral
        """

        norm = self.norm_at_z(z, plaw_index, delta_z, self.m_pivot)
        moment = self.integrate_power_law(norm_scale * norm, mlow, mhigh, log_m_break, n, plaw_index,
                                          break_index=break_index, break_scale=break_scale)

        return moment

    def integrate_power_law(self, norm, m_low, m_high, log_m_break, n, plaw_index, break_index=0, break_scale=1):

        """
        Integrates a power law function of the form

        f(x) = norm * x ^ (n + plaw_index) * (1 + (xc / x)^break_scale )^break_index

        from x = m_low to x = m_high

        :param norm: normalization prefactor
        :param m_low: low bound of integral
        :param m_high: high bound of integral
        :param log_m_break: characteristic break mass (in log) of the power law
        :param n: computes the nth moment
        :param plaw_index: logarithmic slope of power law
        :param break_index: see parameterization above
        :param break_scale: see parameterization above
        :return: the desired integral
        """
        def _integrand(m, m_break, plaw_index, n):

            return norm * m ** (n + plaw_index) * (1 + (m_break / m)**break_scale) ** break_index

        moment = quad(_integrand, m_low, m_high, args=(10**log_m_break, plaw_index, n))[0]

        return moment

    def _mass_function_moment(self, M, dNdM, n, m_low, m_high, order=1):

        """
        :param normalization: dimensions M_sun^-1 Mpc^-3 or M_sun
        :param plaw_index: power law index
        :param N: for the Nth moment
        :return: Nth moment of the mass funciton
        """

        norm, plaw_index = self._fit_norm_index(M, dNdM, order=order)

        if plaw_index == 2 and n==1:
            moment = norm * np.log(m_high * m_low ** -1)

        else:

            moment = self.integrate_power_law(norm, m_low, m_high, 0, n, plaw_index)

        return moment, norm, plaw_index

    def mass_fraction_in_halos(self, z, mlow, mhigh, mlow_global=10**-6):

        """

        :param z: redshift
        :param mlow: the lowest halo mass rendered
        :param mhigh: the largest halo mass rendered
        :param mlow_global: the lowest mass halos that exist (in CDM this is ~1 Earth mass)
        :return: the fraction of dark matter contained in halos between mlow_global and mhigh
        """
        m = np.logspace(np.log10(mlow_global), np.log10(mhigh), 100)
        dndm_total = self.dN_dMdV_comoving(m, z)
        total_mass_in_halos = simps(dndm_total * m, m)

        m_rendered = np.logspace(np.log10(mlow), np.log10(mhigh), 100)
        dndm_rendered = self.dN_dMdV_comoving(m_rendered, z)
        mass_rendered = simps(dndm_rendered * m_rendered, m_rendered)

        return mass_rendered/total_mass_in_halos

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
    # new interpolation is optimized to fit correctly over 4 decades in mass
    l = LensingMassFunction(Cosmology(), 10**7., 10**9, 0.1, 4., 6., use_lookup_table=False)

    fname = './lookup_tables/lookup_sheth99.py'

    with open(fname, 'w') as f:
        f.write('import numpy as np\n')

    write_to_file(fname, 'norm_z_dV', np.round(l._norm_z_dV, 12), 'a')
    write_to_file(fname, 'plaw_index_z', np.round(l._plaw_index_z,2), 'a')

    with open(fname, 'a') as f:
        f.write('z_range = np.array([')
        for zi in l._z_range:
            f.write(str(np.round(zi,2))+', ')
        f.write('])\n\n')

    with open(fname, 'a') as f:
        f.write('delta_z = '+str(np.round(l._delta_z,2))+'\n\n')

# import matplotlib.pyplot as plt
# from pyHalo.Cosmology.cosmology import Cosmology
# cosmo = Cosmology()
# m_pivot = 10**9
# l = LensingMassFunction(cosmo, 10**7, 10**9., 0.1, 4., 6., use_lookup_table=False,
#                         m_pivot=m_pivot)
#
# z = 0.4
# delta_plaw_index = -0.2
# plaw_index = l.plaw_index_z(z) + delta_plaw_index
#
# m = np.logspace(6, 10, 50)
# dndm_colossus = l.dN_dMdV_comoving(m, z) * (m/m_pivot) ** delta_plaw_index
# norm = l.norm_at_z_density(z, plaw_index, m_pivot)
# dndm = norm * m ** plaw_index
# plt.loglog(m, dndm)
# plt.loglog(m, dndm_colossus)
#
# plt.show()

#write_lookup_table()
# import matplotlib.pyplot as plt
#from pyHalo.Cosmology.cosmology import Cosmology
#cosmo = Cosmology()
# l = LensingMassFunction(cosmo, 10**7, 10**9., 0.1, 4., 6., use_lookup_table=False)
# h = cosmo.astropy.h
# galfit = np.loadtxt('HMF_slope.txt', skiprows=1)
# z = galfit[:,0]
# slope = galfit[:,1]
# amp = galfit[:,3]
# a_z = 1/(1 + z)
# amp_colossus = l.dN_dMdV_comoving(10**8, z) * a_z ** 3
# plt.plot(z, amp_colossus, color='r')
# plt.plot(z, amp, color='k')
# plt.show()

#
# m = np.logspace(7, 9, 10)
# logm = np.log10(m)
# z1, z2 = 0., 3.5
# logdndm_1 = np.log10(l.dN_dMdV_comoving(m, z1))
# logdndm_2 = np.log10(l.dN_dMdV_comoving(m, z2))
# import matplotlib.pyplot as plt
# plt.loglog(m, 10**logdndm_1, color='k', label='z=0')
# plt.loglog(m, 10**logdndm_2, color='r', label='z=2.5')
# # plt.legend(fontsize=13)
# # plt.savefig('redshift_evo.pdf')
# plt.show()
# coeffs1 = np.polyfit(logm, logdndm_1, 1)
# coeffs2 = np.polyfit(logm, logdndm_2, 1)
# print(coeffs1)
# print(coeffs2)
#

# # # new interpolation is optimized to fit correctly over 4 decades in mass
# l = LensingMassFunction(Cosmology(), 10**7, 10**9., 0.1, 4., 6.)
# m = np.logspace(7, 14, 100)
# z = 0.
# dndm0 = l.dN_dMdV_comoving(m, z)
#
# z = 0.5
# dndm = l.dN_dMdV_comoving(m, z)
# plt.loglog(m, dndm/dndm0)
#
# z = 1.
# dndm = l.dN_dMdV_comoving(m, z)
# plt.loglog(m, dndm/dndm0)
#
# z = 2.
# dndm = l.dN_dMdV_comoving(m, z)
# plt.loglog(m, dndm/dndm0); plt.show()
# # m = np.logspace(6, 10, 100)
#
# plaw_predicted = l.plaw_index_z(1.4)
# print(plaw_predicted)
#
# plaw = -2.1
# norm = l.norm_at_z_density(0.5, plaw)
# dndm = norm * m ** plaw
# plt.loglog(m, dndm)
#
# plaw = -1.75
# norm = l.norm_at_z_density(0.5, plaw)
# dndm = norm * m ** plaw
# plt.loglog(m, dndm, color='r'); plt.show()

# print(l.norm_at_z_density(0.5, plaw_new))
#
# #print(l_old._norm_dV_interp(0.5))
#
