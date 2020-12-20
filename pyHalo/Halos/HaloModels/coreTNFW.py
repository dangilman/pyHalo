from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.concentration import Concentration
import numpy as np
from pyHalo.Halos.HaloModels.numerical_alphas.coreNFWmodifiedtrunc \
    import InterpCNFWmodtrunc

class coreTNFWSubhalo(Halo):

    def __init__(self, mass, x, y, r3d, mdef, z,
                 sub_flag, lens_cosmo_instance, args, unique_tag):

        raise Exception('the cored TNFW profile is still under development!')
        self.numerical_class = InterpCNFWmodtrunc()
        self._lens_cosmo = lens_cosmo_instance
        self._concentration = Concentration(lens_cosmo_instance)

        super(Halo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag, lens_cosmo_instance, args, unique_tag)

    @property
    def lenstronomy_ID(self):
        return 'NUMERICAL_ALPHA'

    @property
    def lenstronomy_params(self):

        [concentration, rt, core_units_rs] = self.profile_args
        Rs_angle, theta_Rs_nfw = self.lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)

        normalization = self._normalize(Rs_angle, theta_Rs_nfw)

        x, y = np.round(self.x, 4), np.round(self.y, 4)

        Rs_angle = np.round(Rs_angle, 10)

        r_core = np.round(core_units_rs * Rs_angle, 10)

        r_trunc = rt * self.lens_cosmo.cosmo.kpc_proper_per_asec(self.z) ** -1

        kwargs = {'center_x': x, 'center_y': y, 'Rs': Rs_angle,
                  'r_core': r_core, 'norm': normalization, 'r_trunc': r_trunc}

        return kwargs, self.numerical_class

    @property
    def profile_args(self):
        if not hasattr(self, '_profile_args'):

            truncation_radius = self._lens_cosmo.LOS_truncation_rN(self.mass, self.z,
                                                                   self._args['LOS_truncation_factor'])

            concentration = self._concentration.NFW_concentration(self.mass, self.z,
                                                                  logmhm=self._args['log_m_break'],
                                                                  c_scale=self._args['c_scale'],
                                                                  c_power=self._args['c_power'],
                                                                  scatter=self._args['c_scatter'],
                                                                  model=self._args['mc_model'])

            core_units_rs = 0.5

            self._profile_args = [concentration, truncation_radius, core_units_rs]

        return self._profile_args

    def _normalize(self, Rs, theta_Rs_nfw):

        bmin = self.numerical_class._betamin
        taumax = self.numerical_class._tau_max

        trs_corenfw = self.numerical_class(Rs, 0, Rs, bmin, taumax, 1)

        norm = theta_Rs_nfw * trs_corenfw ** -1

        return norm

