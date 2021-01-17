import numpy as np
import numpy.testing as npt
import pytest
from pyHalo.Rendering.SpatialDistributions.nfw_core import ProjectedNFW
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Halos.lens_cosmo import LensCosmo

class TestProjectedNFW(object):

    def setup(self):

        zlens, zsource = 0.5, 1.5
        self.rs = 60
        self.rmax2d = 40
        self.rvir = 350
        self.rcore = 10.
        self.nfw = ProjectedNFW(self.rmax2d, self.rs, self.rcore, self.rvir)

        cosmo = Cosmology()
        self.geometry = Geometry(cosmo, 0.5, 1.5, 6., 'DOUBLE_CONE')
        self.lens_cosmo = LensCosmo(zlens, zsource, cosmo)

    def test_from_keywords_master(self):

        zmin = 0.01
        zmax = 1.98
        log_mlow = 6.
        log_mhigh = 9.
        host_m200 = 10 ** 13.
        LOS_normalization = 2000.
        draw_poisson = False
        log_mass_sheet_min = 7.
        log_mass_sheet_max = 10.
        kappa_scale = 1.
        delta_power_law_index = -0.17
        delta_power_law_index_coupling = 0.5
        cone_opening_angle = 8.
        m_pivot = 10 ** 8
        sigma_sub = 0.1

        keywords_master = {'zmin': zmin,
                      'zmax': zmax,
                      'log_mc': None,
                      'log_mlow': log_mlow,
                      'sigma_sub': sigma_sub,
                      'a_wdm': None, 'b_wdm': None, 'c_wdm': None,
                      'log_mhigh': log_mhigh,
                      'host_m200': host_m200,
                      'host_c': 4.,
                      'host_Rs': 40.,
                      'r_tidal': '0.5Rs',
                      'LOS_normalization': LOS_normalization,
                      'draw_poisson': draw_poisson,
                      'log_mass_sheet_min': log_mass_sheet_min, 'log_mass_sheet_max': log_mass_sheet_max,
                      'kappa_scale': kappa_scale,
                      'delta_power_law_index': delta_power_law_index,
                      'delta_power_law_index_coupling': delta_power_law_index_coupling,
                      'm_pivot': m_pivot,
                      'cone_opening_angle': cone_opening_angle}

        f = ProjectedNFW.from_keywords_master(keywords_master, self.lens_cosmo, self.geometry)
        rendering_radius, Rs, x_core_host, x200 = f.rmax2d_kpc, f._rs_kpc, f.xtidal, f.zmax_units_rs

        npt.assert_equal(rendering_radius, 0.5 * keywords_master['cone_opening_angle'] * self.geometry.kpc_per_arcsec_zlens)
        npt.assert_equal(x_core_host * Rs, 0.5 * Rs)
        npt.assert_equal(x200 * Rs, keywords_master['host_c'] * Rs)
        npt.assert_equal(Rs, keywords_master['host_Rs'])

        x, y, r3 = f.draw(10000)
        r2 = np.hypot(x, y)
        npt.assert_array_less(r2, rendering_radius)

        npt.assert_raises(Exception, ProjectedNFW.from_keywords_master, {'blah': 0.}, self.lens_cosmo, self.geometry)
        npt.assert_raises(Exception, ProjectedNFW.from_keywords_master, {'host_m200': 10**13.,
                                                           'host_c': 4.,'host_Rs': 50.,
                                                           'r_tidal': 'Rss'}, self.lens_cosmo, self.geometry)


    def test_limit(self):

        x, y, r3 = self.nfw.draw(10000)
        r2 = np.hypot(x, y)
        npt.assert_almost_equal(max(r2)/self.rmax2d, 1, 2)

    def test_profile(self):

        x, y, r3 = self.nfw.draw(100000)
        r2 = np.hypot(x, y)
        rbins = np.arange(10, self.rmax2d+5, 5)
        n = []
        for i in range(0, len(rbins) - 1):
            condition = np.logical_and(r2 >= rbins[i], r2 < rbins[i + 1])
            N = np.sum(condition)
            area = np.pi * (rbins[i + 1] ** 2 - rbins[i] ** 2)
            n.append(N / area)

        prof = self.nfw._cnfw_profile._F
        x = rbins / self.rs
        xtidal = self.rcore / self.rs
        true = prof(x, xtidal)
        true *= max(true) ** -1
        n = np.array(n)/max(n)

        for i in range(0, len(true)-1):
            npt.assert_almost_equal(true[i]/n[i], 1, 1)

t = TestProjectedNFW()
t.setup()
t.test_from_keywords_master()

#
# if __name__ == '__main__':
#
#     pytest.main()
