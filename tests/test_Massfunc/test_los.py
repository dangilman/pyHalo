import numpy.testing as npt
from pyHalo.Rendering.parameterizations import BrokenPowerLaw, PowerLaw
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Rendering.Field.PowerLaw.powerlaw import LOSPowerLaw
import numpy as np
from copy import copy
from scipy.integrate import quad

class Test_los(object):

    def setup(self):

        self.mlow, self.mhigh = 10**6, 10**10
        self.zlens = 0.5
        self.zsource = 1.5
        self.angle = 6

        self.lensing_mass_function = LensingMassFunction(Cosmology(), self.mlow, self.mhigh,
                      self.zlens, self.zsource, self.angle, two_halo_term=False, use_lookup_table=True)

        self.args = {'mdef_main': 'TNFW', 'mdef_los': 'TNFW', 'log_mlow': np.log10(self.mlow),
                'log_mhigh': np.log10(self.mhigh), 'power_law_index': -1.9,
                'parent_m200': 10**13, 'parent_c': 3, 'mdef': 'TNFW',
                'break_index': -1.3, 'c_scale': 60, 'c_power': -0.17,
                'r_tidal': '0.4Rs', 'cone_opening_angle': self.angle,
                'fsub': 0.01, 'log_m_break': 0}

    def test_mfunc(self):

        z = 1.05
        mrange = np.logspace(8, np.log10(self.mhigh), 20)
        mfunc = [self.lensing_mass_function.dN_dMdV_comoving(mi, z) for mi in mrange]
        slope, norm = np.polyfit(np.log10(mrange), np.log10(mfunc), 1)

        norm_2 = np.log10(self.lensing_mass_function.norm_at_z_density(z))
        index_2 = self.lensing_mass_function.plaw_index_z(z)

        y1 = 10**norm * mrange ** slope
        y2 = 10**norm_2 * mrange ** index_2

        diff = np.mean(np.absolute(y1-y2)/y2)
        npt.assert_almost_equal(diff, 0.04, 2)

        dz = 0.02
        n_theory = self.lensing_mass_function.\
            integrate_mass_function(z, dz, 10**6, 10**8, 0, 0, 0, n=0)
        norm = self.lensing_mass_function.norm_at_z(z, dz)
        slope = self.lensing_mass_function.plaw_index_z(z)

        def _integrand(m):
            return norm * m ** slope
        n_actual = quad(_integrand, 10**6, 10**8)[0]
        npt.assert_almost_equal(n_actual/n_theory, 1)

        power_law = PowerLaw(power_law_index=slope, log_mlow=6, log_mhigh=8,
                                  normalization=norm)
        power_law.draw_poission = False

        n_rendered = len(power_law.draw())
        npt.assert_(n_rendered/np.ceil(n_actual) == 1)

t = Test_los()
t.setup()
t.test_mfunc()

# if __name__ == '__main__':
#     pytest.main()
