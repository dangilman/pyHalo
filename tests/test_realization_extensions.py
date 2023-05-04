import pytest
from pyHalo.single_realization import SingleHalo
from pyHalo.realization_extensions import RealizationExtensions
from pyHalo.Cosmology.cosmology import Cosmology
from scipy.interpolate import interp1d
import numpy.testing as npt
import numpy as np
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from pyHalo.Halos.tidal_truncation import TruncationRoche, TruncationRN
from pyHalo.Halos.lens_cosmo import LensCosmo

class TestRealizationExtensions(object):

    def test_add_core_collapsed_halos(self):

        halo_mass = 10 ** 8
        x = 0.5
        y = 1.0
        mdef = 'TNFW'
        z = 0.5
        zlens = 0.5
        zsource = 2.0
        astropy_instance = Cosmology().astropy
        truncation_class = TruncationRoche()
        concentration_class = ConcentrationDiemerJoyce(astropy_instance)
        kwargs_halo_model = {'concentration_model_field_halos': concentration_class,
                                    'truncation_model_field_halos': truncation_class,
                                    'kwargs_density_profile': {}}
        single_halo = SingleHalo(halo_mass, x, y, mdef, z, zlens, zsource, subhalo_flag=False,
                 kwargs_halo_model=kwargs_halo_model)
        ext = RealizationExtensions(single_halo)
        new = ext.add_core_collapsed_halos([0], log_slope_halo=3.1, x_core_halo=0.05)
        lens_model_list, zlist, kwargs_halo, _ = new.lensing_quantities()
        npt.assert_string_equal(lens_model_list[0], 'SPL_CORE')
        npt.assert_equal(kwargs_halo[0]['gamma'], 3.1)

    def test_collapse_profile(self):

        halo_mass = 10 ** 9
        x = 0.5
        y = 1.0
        z = 0.5
        zlens = 0.5
        zsource = 2.0
        astropy_instance = Cosmology().astropy
        truncation_class = TruncationRoche()
        concentration_class = ConcentrationDiemerJoyce(astropy_instance, scatter=False)

        kwargs_density_profile_spl_core = {'log_slope_halo': 3.2, 'x_core_halo': 0.05, 'x_match': 'c'}
        kwargs_halo_model = {'concentration_model_field_halos': concentration_class,
                             'truncation_model_field_halos': truncation_class,
                             'kwargs_density_profile': kwargs_density_profile_spl_core}
        realization_1 = SingleHalo(halo_mass, x, y, 'SPL_CORE', z, zlens, zsource, subhalo_flag=False,
                 kwargs_halo_model=kwargs_halo_model)

        kwargs_halo_model_2 = {'concentration_model_field_halos': concentration_class,
                             'truncation_model_field_halos': truncation_class,
                             'kwargs_density_profile': {}}
        realization_2 = SingleHalo(halo_mass, x, y, 'TNFW', z, zlens, zsource, subhalo_flag=False,
                                   kwargs_halo_model=kwargs_halo_model_2)

        ext = RealizationExtensions(realization_2)
        realization_collapsed = ext.add_core_collapsed_halos([0], halo_profile='SPL_CORE', **kwargs_density_profile_spl_core)
        lens_model_list, zlist, kwargs_halo, _ = realization_collapsed.lensing_quantities()
        lens_model_list_1, zlist_1, kwargs_halo_1, _ = realization_1.lensing_quantities()
        npt.assert_string_equal(lens_model_list[0], lens_model_list_1[0])
        npt.assert_equal(zlist[0], zlist_1[0])
        npt.assert_equal(kwargs_halo, kwargs_halo_1)

        kwargs_density_profile_gnfw = {'gamma_inner': 1.9, 'gamma_outer': 2.85, 'x_match': 'c'}
        kwargs_halo_model = {'concentration_model_field_halos': concentration_class,
                             'truncation_model_field_halos': truncation_class,
                             'kwargs_density_profile': kwargs_density_profile_gnfw}
        realization_1 = SingleHalo(halo_mass, x, y, 'GNFW', z, zlens, zsource, subhalo_flag=False,
                                   kwargs_halo_model=kwargs_halo_model)

        kwargs_halo_model_2 = {'concentration_model_field_halos': concentration_class,
                               'truncation_model_field_halos': truncation_class,
                               'kwargs_density_profile': {}}
        realization_2 = SingleHalo(halo_mass, x, y, 'TNFW', z, zlens, zsource, subhalo_flag=False,
                                   kwargs_halo_model=kwargs_halo_model_2)

        ext = RealizationExtensions(realization_2)
        realization_collapsed = ext.add_core_collapsed_halos([0], halo_profile='GNFW', **kwargs_density_profile_gnfw)
        lens_model_list, zlist, kwargs_halo, _ = realization_collapsed.lensing_quantities()
        lens_model_list_1, zlist_1, kwargs_halo_1, _ = realization_1.lensing_quantities()
        npt.assert_string_equal(lens_model_list[0], lens_model_list_1[0])
        npt.assert_equal(zlist[0], zlist_1[0])
        npt.assert_equal(kwargs_halo, kwargs_halo_1)

    def test_collapse_by_mass(self):

        cosmo = Cosmology()
        m_list = 10**np.random.uniform(6, 10, 1000)
        astropy_instance = Cosmology().astropy
        truncation_class = TruncationRoche()
        concentration_class = ConcentrationDiemerJoyce(astropy_instance, scatter=False)
        kwargs_halo_model = {'concentration_model': concentration_class,
                             'truncation_model': truncation_class,
                             'kwargs_density_profile': {}}
        lens_cosmo = LensCosmo(0.5, 2.0, cosmo)
        realization = SingleHalo(m_list[0], 0.0, 0.0, 'TNFW', 0.5, 0.5, 1.5,
                                 subhalo_flag=True, kwargs_halo_model=kwargs_halo_model, lens_cosmo=lens_cosmo)

        for mi in m_list[1:]:
            single_halo = SingleHalo(mi, 0.0, 0.0, 'TNFW', 0.5, 0.5, 1.5,
                                 subhalo_flag=True, kwargs_halo_model=kwargs_halo_model, lens_cosmo=lens_cosmo)
            realization = realization.join(single_halo)
            single_halo = SingleHalo(mi, 0.0, 0.0, 'TNFW', 0.5, 0.5, 1.5,
                                 subhalo_flag=False, kwargs_halo_model=kwargs_halo_model, lens_cosmo=lens_cosmo)
            realization = realization.join(single_halo)

        ext = RealizationExtensions(realization)

        mass_range_subs = [[6, 8], [8, 10]]
        mass_range_field = [[6, 8], [8, 10]]
        p_subs = [0.3, 0.9]
        p_field = [0.8, 0.25]
        kwargs_halo = {'log_slope_halo': 3, 'x_core_halo': 0.05}
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

    def test_collapse_by_mass_redshift(self):

        cosmo = Cosmology()
        m_list = 10 ** np.random.uniform(6, 10, 1000)
        astropy_instance = Cosmology().astropy
        truncation_class = TruncationRoche()
        concentration_class = ConcentrationDiemerJoyce(astropy_instance, scatter=False)
        kwargs_halo_model = {'concentration_model': concentration_class,
                             'truncation_model': truncation_class,
                             'kwargs_density_profile': {}}
        lens_cosmo = LensCosmo(0.5, 2.0, cosmo)
        realization = SingleHalo(m_list[0], 0.0, 0.0, 'TNFW', 0.5, 0.5, 1.5,
                                 subhalo_flag=True, kwargs_halo_model=kwargs_halo_model, lens_cosmo=lens_cosmo)

        for mi in m_list[1:]:
            single_halo = SingleHalo(mi, 0.0, 0.0, 'TNFW', 0.5, 0.5, 1.5,
                                     subhalo_flag=True, kwargs_halo_model=kwargs_halo_model, lens_cosmo=lens_cosmo)
            realization = realization.join(single_halo)
            single_halo = SingleHalo(mi, 0.0, 0.0, 'TNFW', 0.5, 0.5, 1.5,
                                     subhalo_flag=False, kwargs_halo_model=kwargs_halo_model, lens_cosmo=lens_cosmo)
            realization = realization.join(single_halo)

        ext = RealizationExtensions(realization)

        mass_range_subs = [[6, 8], [8, 10]]
        mass_range_field = [[6, 8], [8, 10]]

        custom_func_1 = lambda z: 0.3
        custom_func_2 = lambda z: 0.9
        custom_func_3 = lambda z: 0.8
        custom_func_4 = lambda z: 0.25
        z_eval = 0.5
        p_subs = [custom_func_1, custom_func_2]
        p_field = [custom_func_3, custom_func_4]
        kwargs_sub = [{}, {}]
        kwargs_field = [{}, {}]
        kwargs_halo = {'log_slope_halo': -3, 'x_core_halo': 0.05}
        inds_collapsed = ext.core_collapse_by_mass(mass_range_subs, mass_range_field,
                              p_subs, p_field, kwargs_sub, kwargs_field)
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

        npt.assert_almost_equal(abs(p_subs[0](z_eval) - i_subs_collapsed_1 / i_subs_1), 0, 1)
        npt.assert_almost_equal(abs(p_subs[1](z_eval) - i_subs_collapsed_2 / i_subs_2), 0, 1)
        npt.assert_almost_equal(abs(p_field[0](z_eval) - i_field_collapsed_1 / i_field_1), 0, 1)
        npt.assert_almost_equal(abs(p_field[1](z_eval) - i_field_collapsed_2 / i_field_2), 0, 1)

    def test_add_ULDM_fluctuations(self):

        cosmo = Cosmology()
        astropy_instance = cosmo.astropy
        truncation_class = TruncationRoche()
        concentration_class = ConcentrationDiemerJoyce(astropy_instance, scatter=False)
        kwargs_halo_model = {'concentration_model': concentration_class,
                             'truncation_model': truncation_class,
                             'kwargs_density_profile': {}}
        lens_cosmo = LensCosmo(0.5, 2.0, cosmo)
        single_halo = SingleHalo(10 ** 8, 0.0, 0.0, 'TNFW', 0.5, 0.5, 1.5, 100.0,
                                 subhalo_flag=True, kwargs_halo_model=kwargs_halo_model, lens_cosmo=lens_cosmo)
        ext = RealizationExtensions(single_halo)
        wavelength=0.6 #kpc, correpsonds to m=10^-22 eV for ULDM
        amp_var = 0.04 #in convergence units
        fluc_var = wavelength
        fluc_size = 0.1
        n_cut = 1e4

        # apeture
        x_images = np.array([-0.347, -0.734, -1.096, 0.207])
        y_images = np.array([ 0.964,  0.649, -0.079, -0.148])
        args_aperture = {'x_images':x_images,'y_images':y_images,'aperture':0.25}
        new = ext.add_ULDM_fluctuations(wavelength,amp_var,fluc_size,fluc_var,n_cut,shape='aperture',args=args_aperture)
        _ = new.lensing_quantities()

        #ring
        args_ring = {'rmin':0.95,'rmax':1.05}
        new = ext.add_ULDM_fluctuations(wavelength,amp_var,fluc_size,fluc_var,n_cut,shape='ring',args=args_ring)
        _ = new.lensing_quantities()

        #ellipse
        args_ellipse = {'amin':0.8,'amax':1.7,'bmin':0.4,'bmax':1.2,'angle':np.pi/4}
        new = ext.add_ULDM_fluctuations(wavelength,amp_var,fluc_size,fluc_var,n_cut,shape='ellipse',args=args_ellipse)
        _ = new.lensing_quantities()

    def test_add_pbh(self):

        cosmo = Cosmology()
        astropy_instance = cosmo.astropy
        lens_cosmo = LensCosmo(0.5, 1.5, cosmo)
        truncation_class = TruncationRN(lens_cosmo)
        concentration_class = ConcentrationDiemerJoyce(astropy_instance, scatter=False)
        kwargs_halo_model = {'concentration_model': concentration_class,
                             'truncation_model': truncation_class,
                             'kwargs_density_profile': {}}

        realization = SingleHalo(10**9, 0.9, 0.1, 'TNFW', 0.5, 0.5, 1.5,
                                 subhalo_flag=False, kwargs_halo_model=kwargs_halo_model, lens_cosmo=lens_cosmo)
        zlist = [0.2, 0.4, 0.6]
        rmax = 0.4
        main_halo_coord_x = []
        main_halo_coord_y = []
        for i, zi in enumerate(zlist):
            nhalos = np.random.randint(1, 5)
            _x, _y = [], []
            for j in range(0, nhalos):
                mi = np.random.uniform(8,  9)
                xi = np.random.uniform(-0.2, 0.2)
                yi = np.random.uniform(-0.2, 0.2)
                _x.append(xi)
                _y.append(yi)
                single_halo = SingleHalo(10 ** mi, xi, yi, 'TNFW', zi, 0.5, 1.5, subhalo_flag=False,
                                     kwargs_halo_model=kwargs_halo_model, lens_cosmo=lens_cosmo)
                realization = realization.join(single_halo)
            main_halo_coord_x.append(_x)
            main_halo_coord_y.append(_y)

        lens_model_list_init, _, kwargs_init, _ = realization.lensing_quantities()
        ext = RealizationExtensions(realization)
        mass_fraction = 0.1
        kwargs_mass_function = {'mass_function_type': 'DELTA', 'logM': 5., 'mass_fraction': 0.5}
        fraction_in_halos = 0.9

        _zlist = np.round(np.arange(0.00, 1.02, 0.02), 2)
        x_image = [0.] * len(_zlist)
        y_image = [0.] * len(_zlist)
        cosmo = Cosmology()
        dlist = [cosmo.D_C_transverse(zi) for zi in _zlist]
        x_image_interp_list = [interp1d(dlist, x_image)]
        y_image_interp_list = [interp1d(dlist, y_image)]

        r_array = np.zeros(len(x_image_interp_list))
        r_array[0:] = rmax

        pbh_realization = ext.add_primordial_black_holes(mass_fraction, kwargs_mass_function,
                                                         fraction_in_halos,
                                                         x_image_interp_list,
                                                         y_image_interp_list,
                                                         r_array)

        # import matplotlib.pyplot as plt
        # from pyHalo.single_realization import realization_at_z
        # for j, zi in enumerate(zlist):
        #     real = realization_at_z(pbh_realization, zi)[0]
        #     plt.scatter(real.x, real.y)
        #     plt.scatter(main_halo_coord_x[j], main_halo_coord_y[j], color='r')
        #     plt.show()
        #     a=input('continue')

        for i, halo in enumerate(pbh_realization.halos):

            condition1 = 'PT_MASS' == halo.mdef
            condition2 = 'TNFW' == halo.mdef
            npt.assert_equal(np.logical_or(condition1, condition2), True)

if __name__ == '__main__':
     pytest.main()
