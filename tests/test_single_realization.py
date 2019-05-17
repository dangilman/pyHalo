import numpy.testing as npt
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw, PowerLaw
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.single_realization import Realization
from pyHalo.Massfunc.los import LOSPowerLaw
import pytest
import numpy as np
from copy import copy

class TestSingleRealization(object):

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

        LOS_norm = 1
        self.LOS_norm = LOS_norm
        log_m_break = 0
        draw_poisson = True

        los_mfunc = LOSPowerLaw(self.args, self.lensing_mass_function)
        ind = np.argmin(np.absolute(los_mfunc._redshift_range - self.zlens))
        self.z1 = los_mfunc._redshift_range[ind]
        self.z2 = los_mfunc._redshift_range[ind+2]

        self.delta_z = los_mfunc._redshift_range[1] - los_mfunc._redshift_range[0]
        z1 = self.z1
        z2 = self.z2
        delta_z = self.delta_z

        los_mfunc._parameterization_args.update({'draw_poisson':draw_poisson})
        los_mfunc._parameterization_args.update({'LOS_normalization':LOS_norm})
        los_mfunc._parameterization_args.update({'log_m_break': log_m_break})
        los_mfunc._parameterization_args.update({'break_scale': 1.2})

        plaw_indexes = [self.lensing_mass_function.plaw_index_z(z1),
                        self.lensing_mass_function.plaw_index_z(z2)]
        norms = [LOS_norm*self.lensing_mass_function.norm_at_z(z1, delta_z),
                 LOS_norm*self.lensing_mass_function.norm_at_z(z2, delta_z)]

        pargs = los_mfunc._parameterization_args
        pargs_z1 = copy(los_mfunc._parameterization_args)
        pargs_z2 = copy(los_mfunc._parameterization_args)

        pargs_z1.update({'normalization': norms[0], 'power_law_index': plaw_indexes[0],
                      'draw_poisson': draw_poisson})
        pargs_z2.update({'normalization': norms[1], 'power_law_index': plaw_indexes[1],
                      'draw_poisson': draw_poisson})

        masses_z1 = los_mfunc._draw(norms[0], plaw_indexes[0], pargs, z1)[0]
        masses_z2 = los_mfunc._draw(norms[1], plaw_indexes[1], pargs, z2)[0]

        self.masses_plane_1 = masses_z1
        self.masses_plane_2 = masses_z2

        stuff = los_mfunc()
        masses = stuff[0]
        redshifts = stuff[-1]

        x = y = r2d = r3d = np.zeros_like(masses)
        mdefs = ['NFW']*len(masses)
        mdef_args = {'cone_opening_angle': 6, 'opening_angle_factor': 6}

        self.realization = Realization(masses, x, y, r2d, r3d, mdefs, redshifts,
                                       self.lensing_mass_function, other_params=mdef_args)

    def test_mass_at_z(self):

        m_at_z = self.realization.mass_at_z(self.z1)
        theory_mass_at_z = self.lensing_mass_function.integrate_mass_function(self.z1, self.delta_z,
                              self.mlow, 10**8.5, 0, -1, 1, norm_scale=self.LOS_norm)

        kappa_z = self.realization.convergence_at_z(self.z1, 0)
        area = self.lensing_mass_function.geometry._angle_to_arcsec_area(self.zlens, self.z1)

        sigmacrit_z = self.lensing_mass_function.geometry._lens_cosmo.get_sigmacrit(self.z1)

        kappa_z_theory = m_at_z / area / sigmacrit_z
        npt.assert_almost_equal(kappa_z, kappa_z_theory)

#t = TestSingleRealization()
#t.setup()
#t.test_mass_at_z()

if __name__ == '__main__':
    pytest.main()



