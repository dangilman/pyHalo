from colossus.lss.mass_function import *
from pyHalo.Cosmology.geometry import *
from scipy.interpolate import interp1d
from pyHalo.defaults import *
from colossus.lss.bias import twoHaloTerm
from scipy.integrate import simps


class LensingMassFunction(object):

    """
    This class handles computations pertaining to the halo mass function, including evaluating the normalization and slope
    of the mass function itself, and computing the two halo term.
    """

    def __init__(self, cosmology, zlens, zsource, mlow=None, mhigh=None, cone_opening_angle=None,
                 m_pivot=10**8, mass_function_model='sheth99', use_lookup_table=False,
                 geometry_type=None):

        """

        :param cosmology: An instance of Cosmology (see cosmology.py)
        :param zlens: lens redshift
        :param zsource: source redshift
        :param mlow: low end of mass function
        :param mhigh: high end of mass function
        :param cone_opening_angle: opening angle of lensing volume in arcseconds
        :param m_pivot: pivot mass of the mass function in M_sun
        :param mass_function_model (optional): the halo mass function model, default is Sheth-Tormen
        :param use_lookup_table: Whether to use a precomputed lookup table for the normalization and slope of the mass function
        :param geometry_type: Type of lensing geometry (DOUBLE_CONE, CYLINDER)
        """

        self.cosmo = cosmology

        if mass_function_model is None:
            mass_function_model = realization_default.default_mass_function

        if geometry_type is None:
            geometry_type = lenscone_default.default_geometry

        self.geometry = Geometry(cosmology, zlens, zsource, cone_opening_angle, geometry_type)
        self._mass_function_model = mass_function_model
        self._mlow, self._mhigh = mlow, mhigh

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

    def dN_dMdV_comoving(self, M, z):

        """
        :param M: m (in physical units, no little h)
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (comoving)
        [N * M_sun ^ -1 * Mpc ^ -3]
        """

        h = self.cosmo.h

        # To M_sun / h units
        M_h = M*h
        dndlogM = massFunction(M_h, z, q_out='dndlnM', model=self._mass_function_model)

        dndM_comoving_h = dndlogM / M

        # thee factors of h for the (Mpc/h)^-3 to Mpc^-3 conversion
        dndM_comoving = dndM_comoving_h * h ** 3

        return dndM_comoving

    def component_density(self, component_fraction):

        """

        :param component_fraction: fraction of the comoving matter density
        :return: the comoving dark matter density times component_fraction in comoving units [M * Mpc^-3]

        rho_returned = f * rho_DM
        """

        rho_dV = component_fraction * self.cosmo.rho_dark_matter_crit

        return rho_dV

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

    def plaw_index_z(self, z):

        """

        :param z: redshift
        :return: the logarithmic slope of the halo mass function at redshift z
        """

        idx = self._plaw_interp(z)

        return idx

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

        M = np.logspace(np.log10(mlow), np.log10(mhigh), 20)

        norm, index, norm_scale = [], [], []

        for zi in z_range:

            normi, indexi = self._mass_function_params(M, zi)
            norm.append(normi)
            index.append(indexi)

        return norm, index, z_range, z_range[1] - z_range[0]

    def _mass_function_params(self, M, zstart):

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

        coeffs = np.polyfit(np.log10(M / self.m_pivot), np.log10(dN_dMdV), 1)

        plaw_index = coeffs[0]
        norm_dV = 10 ** coeffs[1]

        return norm_dV, plaw_index

def write_lookup_table():

    def write_to_file(fname, pname, values, mode):
        with open(fname, mode) as f:
            f.write(pname + ' = [')
            for j, val in enumerate(values):

                f.write(str(val)+', ')
            f.write(']\n')

    from pyHalo.Cosmology.cosmology import Cosmology
    l = LensingMassFunction(Cosmology(), 0.1, 4., 10**7., 10**9, 6., use_lookup_table=False)

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

