import numpy as np
import numpy.testing as npt
import pytest
from pyHalo.Rendering.SpatialDistributions.nfw import ProjectedNFW, CNFW
from lenstronomy.Util.analysis_util import azimuthalAverage
import matplotlib.pyplot as plt


class TestProjectedNFW(object):

    def setup_method(self):

        rmax2d_arcsec = 10.0
        rs_kpc = 60
        r200_kpc = 550

        self.arcsec_to_kpc = 6.2
        self.rs_arcsec = rs_kpc / self.arcsec_to_kpc
        self.rmax2d_arcsec = rmax2d_arcsec
        self.r200_arcsec = r200_kpc / self.arcsec_to_kpc
        self.rcore_arcsec = 0.5 * self.rs_arcsec
        self.nfw = ProjectedNFW(self.rmax2d_arcsec, self.rs_arcsec, self.rcore_arcsec,
                                self.r200_arcsec, self.arcsec_to_kpc)

    def test_limit(self):

        x, y, r3 = self.nfw.draw(50000)
        r2_kpc = np.hypot(x, y)
        rmax_kpc = self.rmax2d_arcsec * self.arcsec_to_kpc
        npt.assert_almost_equal(max(r2_kpc)/rmax_kpc, 1, 2)
        rmax_3d_kpc = self.r200_arcsec * self.arcsec_to_kpc
        npt.assert_almost_equal(max(r3)/rmax_3d_kpc,1,1)

    def test_profile(self):

        rbins = np.linspace(0.005*self.rmax2d_arcsec, self.rmax2d_arcsec, 100)

        x_kpc, y_kpc, r3d_kpc = self.nfw.draw(3000000)
        x_arcsec, y_arcsec = x_kpc / self.arcsec_to_kpc, y_kpc / self.arcsec_to_kpc
        r_arcsec = np.hypot(x_arcsec, y_arcsec)
        number_density = []
        for i in range(0, len(rbins)-1):
            cond1 = r_arcsec >= rbins[i]
            cond2 = r_arcsec < rbins[i+1]
            cond = np.logical_and(cond1, cond2)
            area = rbins[i+1]**2 - rbins[i] ** 2
            number_density.append(np.sum(cond)/area)
        number_density = np.array(number_density)

        #plt.loglog(rbins[0:-1], number_density, color='k')

        cnfw_profile = CNFW()
        number_density_true = cnfw_profile.density_2d(rbins, 0.0, self.rs_arcsec, 1.0, self.rcore_arcsec)
        rescale = np.mean(number_density[0:10])/number_density_true[0]
        number_density_true *= rescale
        ratio = []
        for i in range(0, len(rbins[0:-1])):
            ratio.append(number_density_true[i]/number_density[i])
        ratio = np.array(ratio)
        npt.assert_array_less(abs(1-np.sum(ratio)/len(ratio)), 0.05)
        # plt.loglog(rbins, number_density_true, color='b')
        # plt.gca().axvline(self.rcore_arcsec)
        # plt.xlim(0.01, 20)
        # plt.show()

if __name__ == '__main__':
     pytest.main()
