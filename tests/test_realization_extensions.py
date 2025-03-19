import pytest
from pyHalo.single_realization import SingleHalo
from pyHalo.realization_extensions import RealizationExtensions, corr_kappa_with_mask, xi_l, xi_l_to_Pk_l, fit_correlation_multipole
from pyHalo.Cosmology.cosmology import Cosmology
from scipy.interpolate import interp1d
import numpy.testing as npt
import numpy as np
from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
from pyHalo.Halos.tidal_truncation import TruncationRoche, TruncationRN, TruncationGalacticus
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.PresetModels.cdm import CDM
from lenstronomy.LensModel.Profiles.splcore import SPLCORE
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo
from pyHalo.single_realization import Realization

class TestRealizationExtensions(object):

    def test_SIS_injection(self):

        mass = 10 ** 10
        x = 0.0
        y = 0.0
        z = 0.5
        tau = 100
        lens_cosmo = LensCosmo(z, 2.0)
        halo = TNFWFieldHalo.simple_setup(mass, x, y, z, tau, lens_cosmo)
        realization = Realization.from_halos([halo], lens_cosmo, None,
                                             None, None, None,
                                             None, None)
        ext = RealizationExtensions(realization)
        new_realization = ext.SIS_injection(10**11, galaxy_model='SIS')
        npt.assert_equal(new_realization.halos[0].mdef=='TNFW', True)
        new_realization = ext.SIS_injection(10 ** 9, galaxy_model='SIS')
        npt.assert_equal(new_realization.halos[0].mdef == 'SIS', True)

        new_realization = ext.SIS_injection(10 ** 11, galaxy_model='GNFW')
        npt.assert_equal(new_realization.halos[0].mdef == 'TNFW', True)
        new_realization = ext.SIS_injection(10 ** 9, galaxy_model='GNFW')
        npt.assert_equal(new_realization.halos[0].mdef == 'GNFW', True)

        cdm = CDM(0.5, 2.0, mass_threshold_sis=10**9, log_mlow=9, log_mhigh=11,
                  galaxy_model='SIS')
        for halo in cdm.halos:
            npt.assert_string_equal(halo.mdef, 'SIS')
        cdm = CDM(0.5, 2.0, mass_threshold_sis=10 ** 9, log_mlow=9, log_mhigh=11,
                  galaxy_model='GNFW')
        for halo in cdm.halos:
            npt.assert_string_equal(halo.mdef, 'GNFW')

    def test_black_holes(self):

        cdm = CDM(0.5, 2.0, sigma_sub=0.0, LOS_normalization=1.0)
        ext = RealizationExtensions(cdm)
        log10_m_ratio = 0.0
        log10_f = np.log10(0.5)
        mbh = ext.add_black_holes(log10_m_ratio,
                                  log10_f,
                                  log10_mlow_halos_subres=6.0)
        for bh, nfw in zip(mbh.halos, cdm.halos):
            npt.assert_almost_equal(bh.mass/2, nfw.mass)
            npt.assert_almost_equal(bh.redshift, nfw.redshift)

    def test_globular_clusters(self):

        cdm = CDM(0.5, 2.0, sigma_sub=0.01, LOS_normalization=0.1)
        n_halos_cdm = len(cdm.halos)
        ext = RealizationExtensions(cdm)
        log10_mgc_mean = 4.5
        log10_mgc_sigma = 0.2
        rendering_radius_arcsec = 10.0
        gamma_mean = 2.0
        gamma_sigma = 0.2
        gc_concentration_mean = 50
        gc_concentration_sigma = 20
        gc_size_mean = 300
        gc_size_sigma = 150
        gc_surface_mass_density = 1e6
        cdm_with_GCs = ext.add_globular_clusters(
            log10_mgc_mean, log10_mgc_sigma, rendering_radius_arcsec, gamma_mean, gamma_sigma,
            gc_concentration_mean, gc_concentration_sigma, gc_size_mean, gc_size_sigma, gc_surface_mass_density,
            center_x=0, center_y=0
        )
        n_halos_cdm_plus_gcs = len(cdm_with_GCs.halos)
        npt.assert_equal(n_halos_cdm_plus_gcs>n_halos_cdm, True)

        cdm0 = CDM(0.5, 2.0, sigma_sub=0.0, LOS_normalization=0.0)
        ext_onlygc = RealizationExtensions(cdm0)
        gcs = ext_onlygc.add_globular_clusters(
            log10_mgc_mean, log10_mgc_sigma, rendering_radius_arcsec, gamma_mean, gamma_sigma,
            gc_concentration_mean, gc_concentration_sigma, gc_size_mean, gc_size_sigma, gc_surface_mass_density,
            center_x=0, center_y=0
        )
        profile = SPLCORE()
        kpc_per_arcsec = cdm0.lens_cosmo.cosmo.kpc_proper_per_asec(0.5)
        sigma_crit_mpc = cdm0.lens_cosmo.get_sigma_crit_lensing(0.5, 2.0)
        sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2
        mass_total = 0
        for gc in gcs.halos:
            profile_args = gc.profile_args
            rho0 = profile_args[0]
            R = profile_args[1]
            gamma = profile_args[2]
            rc = profile_args[3]
            m_theory = profile.mass_3d(R, rho0, rc, gamma)
            npt.assert_almost_equal(m_theory, gc.mass)
            lenstronomy_params = gc.lenstronomy_params[0][0]
            rho0_arcsec = lenstronomy_params['sigma0'] / lenstronomy_params['r_core']
            R_arcsec = R / kpc_per_arcsec
            gamma = lenstronomy_params['gamma']
            rc_arcsec = lenstronomy_params['r_core']
            m_theory = profile.mass_3d(R_arcsec, rho0_arcsec, rc_arcsec, gamma) * sigma_crit_arcsec
            npt.assert_almost_equal(m_theory, gc.mass)
            mass_total += m_theory

        area = np.pi * (kpc_per_arcsec * rendering_radius_arcsec) ** 2
        sigma = mass_total / area
        npt.assert_almost_equal(sigma / gc_surface_mass_density, 1, 2)

    def test_toSIDM(self):

        cdm = CDM(0.5, 1.5, sigma_sub=0.1, LOS_normalization=0., log_mlow=7.0)
        ext = RealizationExtensions(cdm)
        mass_bin_list = [[6, 10]]
        log10_effective_sigma = [np.log10(1000)]
        log10_subhalo_time_scaling = 0.0
        sidm = ext.toSIDM_from_cross_section(mass_bin_list,
                                             log10_effective_sigma,
                                             log10_subhalo_time_scaling)
        ratio_list = []
        for cdm_halo, sidm_halo in zip(cdm.halos, sidm.halos):

            rs = cdm_halo.nfw_params[1]
            kwargs_nfw = cdm_halo.lenstronomy_params[0]
            kwargs_sidm = sidm_halo.lenstronomy_params[0]
            npt.assert_equal(cdm_halo.c, sidm_halo.c)
            for kw in kwargs_nfw:
                kw['center_x'] = 0.0
                kw['center_y'] = 0.0
            for kw in kwargs_sidm:
                kw['center_x'] = 0.0
                kw['center_y'] = 0.0

            _, rt_kpc = cdm_halo.profile_args
            tau = rt_kpc / cdm_halo.nfw_params[1]
            x = np.linspace(min(0.01 * tau, 0.01), cdm_halo.c, 5000)
            r = x * rs
            rho_nfw = cdm_halo.density_profile_3d_lenstronomy(r)
            rho_sidm = sidm_halo.density_profile_3d_lenstronomy(r)
            mass_nfw = np.trapz(4 * np.pi * r ** 2 * rho_nfw, r)
            mass_sidm = np.trapz(4 * np.pi * r ** 2 * rho_sidm, r)
            ratio = mass_sidm / mass_nfw
            ratio_list.append(ratio)
        # import matplotlib.pyplot as plt
        # plt.hist(ratio_list,bins=50,range=(0.9,1.25))
        # plt.show()
        npt.assert_array_less(abs(np.median(ratio_list)-1), 0.05)
        npt.assert_array_less(np.std(ratio_list), 0.1)

        cdm = CDM(0.5, 1.5, sigma_sub=0.0, LOS_normalization=1., log_mlow=7.0)
        ext = RealizationExtensions(cdm)
        mass_bin_list = [[6, 10]]
        log10_effective_sigma = [np.log10(1000)]
        log10_subhalo_time_scaling = 0.0
        sidm = ext.toSIDM_from_cross_section(mass_bin_list,
                                             log10_effective_sigma,
                                             log10_subhalo_time_scaling)
        ratio_list = []
        for cdm_halo, sidm_halo in zip(cdm.halos, sidm.halos):

            rs = cdm_halo.nfw_params[1]
            kwargs_nfw = cdm_halo.lenstronomy_params[0]
            kwargs_sidm = sidm_halo.lenstronomy_params[0]
            npt.assert_equal(cdm_halo.c, sidm_halo.c)
            for kw in kwargs_nfw:
                kw['center_x'] = 0.0
                kw['center_y'] = 0.0
            for kw in kwargs_sidm:
                kw['center_x'] = 0.0
                kw['center_y'] = 0.0

            _, rt_kpc = cdm_halo.profile_args
            tau = rt_kpc / cdm_halo.nfw_params[1]
            x = np.linspace(min(0.01 * tau, 0.01), cdm_halo.c, 5000)
            r = x * rs
            rho_nfw = cdm_halo.density_profile_3d_lenstronomy(r)
            rho_sidm = sidm_halo.density_profile_3d_lenstronomy(r)
            mass_nfw = np.trapz(4 * np.pi * r ** 2 * rho_nfw, r)
            mass_sidm = np.trapz(4 * np.pi * r ** 2 * rho_sidm, r)
            ratio = mass_sidm / mass_nfw
            ratio_list.append(ratio)
        # import matplotlib.pyplot as plt
        # plt.hist(ratio_list,bins=50,range=(0.9,1.25))
        # plt.show()
        npt.assert_array_less(abs(np.median(ratio_list) - 1), 0.05)
        npt.assert_array_less(np.std(ratio_list), 0.05)

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

        for i, halo in enumerate(pbh_realization.halos):

            condition1 = 'PT_MASS' == halo.mdef
            condition2 = 'TNFW' == halo.mdef
            npt.assert_equal(np.logical_or(condition1, condition2), True)

if __name__ == '__main__':
      pytest.main()

# class TestCorrelationComputation(object):
#
#     def setup_method(self):
#         pass
#
#     def test_corr_kappa_with_mask(self):
#         npix = 500
#         window_size = 4
#         delta_pix = window_size/npix
#         mu = np.linspace(-1, 1, 100)
#         r = np.logspace(np.log10(2*10**(-2)), -0.3, num=100, endpoint=True)
#         _R = np.linspace(-window_size/2, window_size/2, npix)
#         XX, YY = np.meshgrid(_R, _R)
#
#         def kappa_GRF(delta_pix, num_pix, alpha):
#             #Generating Gaussian random field kappa map
#             noise = (delta_pix**2)*np.fft.fft2(np.random.normal(size=(num_pix,num_pix)))
#             fftind = 2.0*np.pi*np.fft.fftfreq(num_pix, d=delta_pix)
#             kxi,kyi = np.meshgrid(fftind,fftind)
#             kvnorm2 = kxi**2 + kyi**2 + 1e-10
#             amplitude = np.sqrt((kvnorm2*delta_pix**2)**(-alpha/2))
#             kappa = np.fft.ifft2(amplitude*noise)/delta_pix**2
#
#             return kappa.real - np.mean(kappa.real)
#
#         alpha = 1
#         kappa = kappa_GRF(delta_pix, npix, alpha)
#
#         corr = corr_kappa_with_mask(kappa, window_size, r, mu, apply_mask = False, r_min = 0, r_max = None, normalization = False)
#         corr_mask = corr_kappa_with_mask(kappa, window_size, r, mu, apply_mask = True, r_min = 0.5, r_max = None, normalization = False)
#         corr_mask_ann = corr_kappa_with_mask(kappa, window_size, r, mu, apply_mask = True, r_min = 0.5, r_max = 1.5, normalization = False)
#         corr_norm = corr_kappa_with_mask(kappa, window_size, r, mu, apply_mask = False, r_min = 0, r_max = None, normalization = True)
#         corr_mask_norm = corr_kappa_with_mask(kappa, window_size, r, mu, apply_mask = True, r_min = 0.5, r_max = None, normalization = True)
#         corr_mask_ann_norm = corr_kappa_with_mask(kappa, window_size, r, mu, apply_mask = True, r_min = 0.5, r_max = 1.5, normalization = True)
#
#         xi_0_real = delta_pix**(2-alpha)/(2*np.pi*r)
#
#         mu_grid = np.tile(mu, (r.shape[0], 1))
#         T_l_grid = eval_chebyt(0, mu_grid)
#         xi_l_grid = np.transpose([xi_0_real] *mu.shape[0])
#
#         corr_real = xi_l_grid*T_l_grid
#         corr_real_norm = np.linalg.norm(corr_real, 1)*corr_real
#
#         npt.assert_array_almost_equal(corr_real, corr, decimal=2)
#         npt.assert_array_almost_equal(corr_real, corr_mask, decimal=2)
#         npt.assert_array_almost_equal(corr_real, corr_mask_ann, decimal=2)
#         npt.assert_array_almost_equal(corr_real_norm,corr_norm, 1)
#         npt.assert_array_almost_equal(corr_real_norm,corr_mask_norm, 1)
#         npt.assert_array_almost_equal(corr_real_norm,corr_mask_ann_norm, 1)
#
#     def test_xi_l(self):
#         mu = np.linspace(-1, 1, 100)
#         r = np.logspace(-3, -0.3, num=100, endpoint=True)
#         xi_0_real = np.ones(r.shape[0])
#         corr = np.ones((r.shape[0], mu.shape[0]))
#         r, xi_0 = xi_l(0, corr, r, mu)
#         npt.assert_almost_equal(xi_0_real, xi_0)
#
#     def test_xi_l_to_Pk_l(self):
#         l = 0
#         x = np.logspace(-3, 3, num=60, endpoint=False)
#         F = 1 / (1 + x*x)**1.5
#         y, G_Hankel = xi_l_to_Pk_l(x, F, l = 0)
#         G = (2*np.pi*(-1j)**l) * np.exp(-y)  # this is the actual Hankel transform of the function F.
#         npt.assert_almost_equal(G, G_Hankel)
#
#     def test_fit_correlation_multipole(self):
#         r = np.logspace(-1, 2, num=100, endpoint=True)
#         As = 5
#         n = -3
#         r_min, r_max = 0, 50
#         r_pivot = (r_min + r_max)/2
#         func_real = As*(r/r_pivot)**n
#
#         As_fit, n_fit = fit_correlation_multipole(r, func_real, r_min, r_max)
#         npt.assert_array_almost_equal(As, As_fit)
#         npt.assert_array_almost_equal(n, n_fit)

#

