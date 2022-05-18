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

        def p_short(*args, **kwargs):
            return 1e-9
        def p_long(*args, **kwargs):
            return 1.0
        class cross_section_class(object):
                pass


        single_halo = SingleHalo(10 ** 8, 0.5, -0.1, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=True)
        ext = RealizationExtensions(single_halo)
        vfunc = lambda x: 4 / np.sqrt(3.1459)

        indexes = ext.find_core_collapsed_halos(p_short, cross_section_class)
        npt.assert_equal(True, 0 in indexes)
        indexes = ext.find_core_collapsed_halos(p_long, cross_section_class)
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

    def test_add_ULDM_fluctuations(self):

        single_halo = SingleHalo(10 ** 8, 0.5, -0.1, 'TNFW', 0.5, 0.5, 1.5, subhalo_flag=True)
        ext = RealizationExtensions(single_halo)
        wavelength=0.6 #kpc, correpsonds to m=10^-22 eV for ULDM
        amp_var = 0.04 #in convergence units
        fluc_var = wavelength
        n_cut = 1e4

        # apeture
        x_images = np.array([-0.347, -0.734, -1.096, 0.207])
        y_images = np.array([ 0.964,  0.649, -0.079, -0.148])
        args_aperture = {'x_images':x_images,'y_images':y_images,'aperture':0.25}

        ext.add_ULDM_fluctuations(wavelength,amp_var,fluc_var,shape='aperture',args=args_aperture, n_cut=n_cut)

        #ring
        args_ring = {'rmin':0.95,'rmax':1.05}
        ext.add_ULDM_fluctuations(wavelength,amp_var,fluc_var,shape='ring',args=args_ring, n_cut=n_cut)

        #ellipse
        args_ellipse = {'amin':0.8,'amax':1.7,'bmin':0.4,'bmax':1.2,'angle':np.pi/4}
        ext.add_ULDM_fluctuations(wavelength,amp_var,fluc_var,shape='ellipse',args=args_ellipse, n_cut=n_cut)

    def test_add_pbh(self):

        kwargs_halo = {'c_scatter': False}
        realization = SingleHalo(10 ** 9, 0., 0., 'TNFW', 0.1, 0.5, 1.5, subhalo_flag=False, kwargs_halo=kwargs_halo)
        zlist = [0.2, 0.4, 0.6]
        rmax = 0.3
        for i, zi in enumerate(zlist):
            theta = np.random.uniform(0., 2*np.pi)
            r = np.random.uniform(0, rmax**2)**0.5
            xi, yi = np.cos(theta) *r, np.sin(theta) * r
            mi = np.random.uniform(8,  9)
            single_halo = SingleHalo(10 ** mi, xi, yi, 'TNFW', zi, 0.5, 1.5, subhalo_flag=False, kwargs_halo=kwargs_halo)
            realization = realization.join(single_halo)

        lens_model_list_init, _, kwargs_init, _ = realization.lensing_quantities()
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

        lens_model_list, _, kwargs, _ = pbh_realization.lensing_quantities()

        for i, halo in enumerate(pbh_realization.halos):
            r2d = np.hypot(halo.x, halo.y)
            npt.assert_equal(r2d <= np.sqrt(2) * rmax, True)
            condition1 = 'PT_MASS' == halo.mdef
            condition2 = 'TNFW' == halo.mdef
            npt.assert_equal(np.logical_or(condition1, condition2), True)

if __name__ == '__main__':
     pytest.main()
