import pytest
from pyHalo.single_realization import SingleHalo
from pyHalo.realization_extensions import RealizationExtensions
from pyHalo.Cosmology.cosmology import Cosmology
from scipy.interpolate import interp1d
import numpy.testing as npt
import numpy as np

class TestRealizationExtensions(object):

    def test_core_collapsed_halo(self):

        single_halo = SingleHalo(10 ** 8, 0.5, -0.1, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=True)
        ext = RealizationExtensions(single_halo)
        new = ext.add_core_collapsed_halos([0], log_slope_halo=3., x_core_halo=0.05)
        lens_model_list = new.lensing_quantities()[0]
        npt.assert_string_equal(lens_model_list[0], 'SPL_CORE')

    def core_collapsed_halos(self):

        def timescalefunction_short(rhos, rs, v):
            return 1e-9
        def timescalefunction_long(rhos, rs, v):
            return 1e9

        single_halo = SingleHalo(10 ** 8, 0.5, -0.1, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=True)
        ext = RealizationExtensions(single_halo)
        vfunc = lambda x: 4 / np.sqrt(3.1459)

        indexes = ext.find_core_collapsed_halos(timescalefunction_short, vfunc)
        npt.assert_equal(True, 0 in indexes)
        indexes = ext.find_core_collapsed_halos(timescalefunction_long, vfunc)
        npt.assert_equal(False, 0 in indexes)

    def test_collapse_by_mass(self):

        cosmo = Cosmology()
        m_list = 10**np.random.uniform(6, 10, 1000)
        realization = SingleHalo(m_list[0], 0.5, -0.1, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=True,
                                 cosmo=cosmo)
        for mi in m_list[1:]:
            single_halo = SingleHalo(mi, 0.5, -0.1, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=True, cosmo=cosmo)
            realization = realization.join(single_halo)
            single_halo = SingleHalo(mi, 0.5, -0.1, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=False, cosmo=cosmo)
            realization = realization.join(single_halo)

        ext = RealizationExtensions(realization)

        mass_range_subs = [[6, 8], [8, 10]]
        mass_range_field = [[6, 8], [8, 10]]
        p_subs = [0.3, 0.9]
        p_field = [0.8, 0.25]
        kwargs_halo = {'log_slope_halo': -3, 'x_core_halo': 0.05}
        inds_collapsed = ext.core_collapse_by_mass(mass_range_subs, mass_range_field,
                              p_subs, p_field)
        realization_collapsed = ext.add_core_collapsed_halos(inds_collapsed, **kwargs_halo)


        i_subs_collapsed_1 = 0
        i_subs_1 = 0
        i_field_collapsed_1 = 0
        i_field_1 = 0
        i_subs_collapsed_2 = 0
        i_subs_2 = 0
        i_field_collapsed_2 = 0
        i_field_2 = 0
        for halo in realization_collapsed.halos:
            print(halo.mdef)
            if halo.is_subhalo:
                if halo.mass < 10 ** 8:
                    i_subs_1 += 1
                    if halo.mdef == 'SPL_CORE':
                        i_subs_collapsed_1 += 1
                else:
                    i_subs_2 += 1
                    if halo.mdef == 'SPL_CORE':
                        i_subs_collapsed_2 += 1
            else:
                if halo.mass < 10 ** 8:
                    i_field_1 += 1
                    if halo.mdef == 'SPL_CORE':
                        i_field_collapsed_1 += 1
                else:
                    i_field_2 += 1
                    if halo.mdef == 'SPL_CORE':
                        i_field_collapsed_2 += 1

        npt.assert_almost_equal(abs(p_subs[0] - i_subs_collapsed_1 / i_subs_1), 0, 1)
        npt.assert_almost_equal(abs(p_subs[1] - i_subs_collapsed_2 / i_subs_2), 0, 1)
        npt.assert_almost_equal(abs(p_field[0] - i_field_collapsed_1 / i_field_1), 0, 1)
        npt.assert_almost_equal(abs(p_field[1] - i_field_collapsed_2 / i_field_2), 0, 1)

    def test_add_pbh(self):

        realization = SingleHalo(10 ** 8, 0., -0.1, 'TNFW', 0.02, 0.5, 1.5, subhalo_flag=False)
        zlist = np.arange(0.02, 1., 0.02)
        rmax = 0.3
        for i, zi in enumerate(zlist):
            theta = np.random.uniform(0., 2*np.pi)
            r = np.random.uniform(0, rmax**2)**0.5
            xi, yi = np.cos(theta) *r, np.sin(theta) * r
            mi = np.random.uniform(7,  8)
            single_halo = SingleHalo(10 ** mi, xi, yi, 'TNFW', zi, 0.5, 1.5, subhalo_flag=False)
            realization = realization.join(single_halo)

        ext = RealizationExtensions(realization)
        mass_fraction = 0.1
        kwargs_mass_function = {'mass_function_type': 'DELTA', 'logM': 5., 'mass_fraction': 0.5}
        fraction_in_halos = 0.5

        zlist = np.arange(0.00, 1.02, 0.02)
        x_image = [0.] * len(zlist)
        y_image = [0.] * len(zlist)
        cosmo = Cosmology()
        dlist = [cosmo.D_C_transverse(zi) for zi in zlist]
        x_image_interp_list = [interp1d(dlist, x_image)]
        y_image_interp_list = [interp1d(dlist, y_image)]

        pbh_realization = ext.add_primordial_black_holes(mass_fraction, kwargs_mass_function,
                                                         fraction_in_halos,
                                                         x_image_interp_list,
                                                         y_image_interp_list,
                                                         rmax)
        for i, halo in enumerate(pbh_realization.halos):
            r2d = np.hypot(halo.x, halo.y)
            npt.assert_equal(r2d <= np.sqrt(2) * rmax, True)
            condition1 = 'PT_MASS' == halo.mdef
            condition2 = 'TNFW' == halo.mdef
            npt.assert_equal(np.logical_or(condition1, condition2), True)

if __name__ == '__main__':
     pytest.main()
