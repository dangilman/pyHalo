from colossus.lss.mass_function import *
from pyHalo.Cosmology.geometry import *
from scipy.interpolate import interp1d
from pyHalo.defaults import *
from colossus.lss.bias import twoHaloTerm

class LensingMassFunction(object):

    def __init__(self,cosmology,mlow,mhigh,zlens,zsource,cone_opening_angle,
                 delta_theta_lens=None, model_kwargs={'model':'sheth99'},
                 use_lookup_table=True, two_halo_term = True):

        if delta_theta_lens is None:
            delta_theta_lens = cone_opening_angle

        self._cosmo = cosmology
        self.geometry = Geometry(cosmology, zlens, zsource, delta_theta_lens, cone_opening_angle)
        self._model_kwargs = model_kwargs
        self._mlow, self._mhigh = mlow, mhigh
        self._two_halo_term = two_halo_term

        self._norms_z_dV, self._plaw_indexes_z, self._log_mbin = [], [], []

        if use_lookup_table:

            if model_kwargs['model'] == 'sheth99':
                from pyHalo.Cosmology.lookup_tables import lookup_sheth99 as table
            else:
                raise ValueError('lookup table '+model_kwargs['model']+' not found.')

            norm_z_dV_bins = table.norm_z_dV_bins
            plaw_index_z_bins = table.plaw_index_z_bins
            z_range = table.z_range
            delta_z = table.delta_z
            mbins = table.mbins

        else:
            # list ordering is by mass, with sublists consisting of different redshifts
            norm_z_dV_bins, plaw_index_z_bins, z_range, delta_z, mbins = self._build(mlow, mhigh, zsource, zlens)

            self.norm_z_dV_bins = norm_z_dV_bins
            self.plaw_index_z_bins = plaw_index_z_bins
            self._z_range = z_range
            self._delta_z = delta_z
            self.mbins = mbins

        for i, mass_bin in enumerate(mbins):
            norm_in_bin = interp1d(z_range, norm_z_dV_bins[i])
            plaw_index_in_bin = interp1d(z_range, plaw_index_z_bins[i])
            self._norms_z_dV.append(norm_in_bin)
            self._plaw_indexes_z.append(plaw_index_in_bin)
            self._log_mbin.append([np.log10(mass_bin[0]), np.log10(mass_bin[1])])

        self._delta_z = delta_z

        self._z_range = z_range

    def norm_at_z_density(self, mscale, z):

        bin_index = self._get_mass_bin_index(np.log10(mscale))

        norm = self._norms_z_dV[bin_index](z)

        return norm

    def plaw_index_z(self, mscale, z):

        bin_index = self._get_mass_bin_index(np.log10(mscale))

        idx = self._plaw_indexes_z[bin_index](z)

        return idx

    def _get_mass_bin_index(self, logmscale):

        for i, bin in enumerate(self._log_mbin):

            if logmscale >= bin[0] and logmscale < bin[1]:
                return i

        else:
            for i, bin in enumerate(self._log_mbin):

                if logmscale == bin[1]:
                    return i

        raise Exception('mass scale '+str(logmscale)+' not in precomputed mass bins.')

    def norm_at_z(self, mscale, z, delta_z):

        norm_dV = self.norm_at_z_density(mscale, z)

        dV = self.geometry.volume_element_comoving(z, self.geometry._zlens, delta_z)

        return norm_dV * dV

    def norm_at_z_biased(self, mscale, z, M_halo, delta_z):

        delta_R = self.geometry.delta_R_fromz(z)

        norm_unbiased = self.norm_at_z(mscale, z, delta_z)

        if self._two_halo_term is False:

            return norm_unbiased

        if delta_R >= 50 or delta_R < 0.5:
            return norm_unbiased
        else:

            delta_R = max(delta_R, 1e-4)
            boost = 1 + self.twohaloterm(delta_R,M_halo,z)

            return norm_unbiased * boost

    def twohaloterm(self, r, M, z, mdef='200c'):

        h = self._cosmo.h
        M_h = M * h
        r_h = r * h

        rho_2h = twoHaloTerm(r_h, M_h, z, mdef=mdef) * self._cosmo._colossus_cosmo.rho_m(z) ** -1

        return rho_2h * h ** -2

    def _build(self, mlow, mhigh, zsource, log_mass_resolution = 0.25, sublog_mass_resolution = 0.025):

        #nsteps = (zsource - 2*default_zstart) * self.geometry._min_delta_z ** -1
        nsteps = 250
        z_range = np.linspace(default_zstart, zsource - default_zstart, nsteps)
        delta_z = z_range[1] - z_range[0]
        norm_bins, index_bins = [], []

        n_mass_bins = max(10, int((np.log10(mhigh) - np.log10(mlow)) * log_mass_resolution ** -1))
        subn_mass_bins = max(10, int(log_mass_resolution / sublog_mass_resolution))

        mrange = np.linspace(np.log10(mlow), np.log10(mhigh), n_mass_bins)
        mbins = []

        for k in range(0, len(mrange) - 1):
            mbins.append([10**mrange[k], 10**mrange[k + 1]])

        for k in range(0, len(mrange) - 1):

            M = np.logspace(np.log10(mbins[k][0]), np.log10(mbins[k][1]), subn_mass_bins)

            norm, index = [], []

            for i, zi in enumerate(z_range):

                _, normi, indexi = self._mass_function_params(M, M[0], M[-1], zi)

                norm.append(normi)
                index.append(indexi)

            norm_bins.append(norm)
            index_bins.append(index)

        return norm_bins, index_bins, z_range, delta_z, mbins

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

        #dN_dM = self.dN_dM(M, zstart, zend, z_lens)
        dN_dMdV = self.dN_dMdV_comoving(M, zstart)

        #N_objects, norm, plaw_index = self._mass_function_moment(M, dN_dM, 0, mlow, mhigh)
        # returns x / mpc ^ 3
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

        return h ** 3 * massFunction(M_h, z, q_out='dndlnM', **self._model_kwargs) * M_h ** -1

    def dN_dV_comoving_deltaFunc(self, M, z, component_fraction):

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

    fname = './lookup_tables/lookup_sheth99.py'

    with open(fname, 'w') as f:
        f.write('import numpy as np\n')

    write_to_file(fname, 'norm_z_dV_bins', l.norm_z_dV_bins, 'a')
    write_to_file(fname, 'plaw_index_z_bins', l.plaw_index_z_bins, 'a')
    write_to_file(fname, 'mbins', l.mbins, 'a')

    with open(fname, 'a') as f:
        f.write('z_range = np.array([')
        for zi in l._z_range:
            f.write(str(zi)+', ')
        f.write('])\n\n')

    with open(fname, 'a') as f:
        f.write('delta_z = '+str(l._delta_z)+'\n\n')
