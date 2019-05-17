import numpy.testing as npt
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw, PowerLaw
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Massfunc.los import LOSPowerLaw
import pytest
import numpy as np
from copy import copy

class Test_los(object):

    def setup(self):

        self.mlow, self.mhigh = 10**6, 10**10
        self.zlens = 0.5
        self.zsource = 1.5
        self.angle = 6

        self.lensing_mass_function = LensingMassFunction(Cosmology(), self.mlow, self.mhigh,
                      self.zlens, self.zsource, self.angle, two_halo_term=False)

        self.args = {'mdef_main': 'TNFW', 'mdef_los': 'TNFW', 'log_mlow': np.log10(self.mlow),
                'log_mhigh': np.log10(self.mhigh), 'power_law_index': -1.9,
                'parent_m200': 10**13, 'parent_c': 3, 'mdef': 'TNFW',
                'break_index': -1.3, 'c_scale': 60, 'c_power': -0.17,
                'r_tidal': '0.4Rs', 'cone_opening_angle': self.angle,
                'fsub': 0.01, 'log_m_break': 0}

    def test_draw(self):

        z1_values = [0.21, 0.67]
        z2_values = [0.23, 0.69]
        LOS_norm = 1e+4

        for log_m_break in [0, 8]:

            for z1, z2 in zip(z1_values, z2_values):

                los_mfunc = LOSPowerLaw(self.args, self.lensing_mass_function)

                delta_z = z2 - z1

                los_mfunc._redshift_range = np.array([z1, z2])
                los_mfunc._parameterization_args.update({'draw_poisson':False})
                los_mfunc._parameterization_args.update({'LOS_normalization':LOS_norm})
                los_mfunc._parameterization_args.update({'log_m_break': log_m_break,
                                                         'break_scale': 1.2})

                plaw_indexes = [self.lensing_mass_function.plaw_index_z(z1),
                                self.lensing_mass_function.plaw_index_z(z2)]
                norms = [LOS_norm*self.lensing_mass_function.norm_at_z(z1, delta_z),
                         LOS_norm*self.lensing_mass_function.norm_at_z(z2, delta_z)]

                pargs = los_mfunc._parameterization_args
                pargs_z1 = copy(los_mfunc._parameterization_args)
                pargs_z2 = copy(los_mfunc._parameterization_args)

                pargs_z1.update({'normalization': norms[0], 'power_law_index': plaw_indexes[0],
                              'draw_poisson': False})
                pargs_z2.update({'normalization': norms[1], 'power_law_index': plaw_indexes[1],
                              'draw_poisson': False})

                mfunc_z1 = BrokenPowerLaw(**pargs_z1)
                mfunc_z2 = BrokenPowerLaw(**pargs_z2)

                masses_explicit_z1 = mfunc_z1.draw()
                masses_explicit_z2 = mfunc_z2.draw()

                Nhalos_z1_explicit = len(masses_explicit_z1)
                Nhalos_z2_explicit = len(masses_explicit_z2)

                masses_z1 = los_mfunc._draw(norms[0], plaw_indexes[0], pargs, z1)[0]
                masses_z2 = los_mfunc._draw(norms[1], plaw_indexes[1], pargs, z2)[0]
                masses = los_mfunc()[0]

                Nhalos_z1 = len(masses_z1)
                Nhalos_z2 = len(masses_z2)

                npt.assert_(Nhalos_z1 == Nhalos_z1_explicit)
                npt.assert_(Nhalos_z2 == Nhalos_z2_explicit)

                npt.assert_(len(masses) == Nhalos_z1 + Nhalos_z2)

                theory_mass = self.lensing_mass_function.integrate_mass_function(z1, delta_z, self.mlow,
                                              self.mhigh, 0, -1, 1, norm_scale=LOS_norm)

                npt.assert_almost_equal(np.absolute(theory_mass / np.sum(masses_z1)), 1, decimal= 1)

if __name__ == '__main__':
    pytest.main()
