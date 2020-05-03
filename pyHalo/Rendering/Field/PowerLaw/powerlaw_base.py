from pyHalo.Rendering.Field.base import LOSBase
from pyHalo.defaults import *
from pyHalo.Rendering.MassFunctions.broken_powerlaw import BrokenPowerLaw
from copy import deepcopy
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, \
    integrate_power_law_analytic

class PowerLawBase(LOSBase):

    def render_masses(self, zi, delta_zi, aperture_radius):

        volume_element_comoving = self.geometry.volume_element_comoving(zi, delta_zi, aperture_radius)

        norm = self.normalization(zi, delta_zi, self._zlens, self.halo_mass_function, self.rendering_args,
                                  volume_element_comoving)

        args = deepcopy(self.rendering_args)

        plaw_index = self.halo_mass_function.plaw_index_z(zi)
        args.update({'normalization': norm})
        args.update({'power_law_index': plaw_index})
        mfunc = BrokenPowerLaw(**args)

        m = mfunc.draw()

        return m

    def negative_kappa_sheets_theory(self):

        kwargs_mass_sheets = self.keys_convergence_sheets

        log_mass_sheet_correction_min, log_mass_sheet_correction_max = \
            kwargs_mass_sheets['log_mass_sheet_min'], kwargs_mass_sheets['log_mass_sheet_max']

        kappa_scale = kwargs_mass_sheets['kappa_scale']

        kappa_sheets = []

        m_low, m_high = 10 ** log_mass_sheet_correction_min, 10 ** log_mass_sheet_correction_max

        log_m_break = self.rendering_args['log_m_break']
        break_index = self.rendering_args['break_index']
        break_scale = self.rendering_args['break_scale']

        moment = 1

        if log_m_break == 0 or log_m_break / log_mass_sheet_correction_min < 0.01:
            use_analytic = True
        else:
            use_analytic = False

        for z, delta_z in zip(self.lens_plane_redshifts, self.delta_zs):

            volume_element_comoving = self.geometry.volume_element_comoving(z, delta_z, None)
            norm = self.normalization(z, delta_z, self.geometry._zlens, self.halo_mass_function,
                                      self.rendering_args, volume_element_comoving)
            plaw_index = self.halo_mass_function.plaw_index_z(z)

            if use_analytic:
                mass = integrate_power_law_analytic(norm, m_low, m_high, moment, plaw_index)
            else:
                mass = integrate_power_law_quad(norm, m_low, m_high, log_m_break, moment,
                                                plaw_index, break_index, break_scale)

            negative_kappa = -1 * kappa_scale * mass / self.lens_cosmo.sigma_crit_mass(z, self.geometry)

            kappa_sheets.append(negative_kappa)

        return kappa_sheets, self.lens_plane_redshifts

    def normalization(self, z, delta_z, zlens, lensing_mass_function_class, rendering_args, volume_element_comoving):

        boost = self.two_halo_boost(z, delta_z, rendering_args['parent_m200'], zlens, lensing_mass_function_class)

        norm_dV = rendering_args['LOS_normalization'] * boost * lensing_mass_function_class.norm_at_z_density(z)

        return norm_dV * volume_element_comoving

    @property
    def keys_convergence_sheets(self):

        args_convergence_sheets = {}
        required_keys = ['log_mass_sheet_min', 'log_mass_sheet_max', 'kappa_scale']

        for key in required_keys:
            if key not in self.rendering_args.keys():
                raise Exception('When specifying mass function type POWER_LAW, must provide '
                                'key word arguments log_mass_sheet_min, log_mass_sheet_max, and kappa_scale.'
                                'These key words specify the halo mass range used to add the convergence correction.')

            args_convergence_sheets[key] = self.rendering_args[key]

        return args_convergence_sheets

    @staticmethod
    def keyword_parse(args, lens_mass_function):

        args_mfunc = {}
        required_keys = ['zmin', 'zmax', 'log_m_break', 'log_mlow_los',
                         'log_mhigh_los', 'parent_m200', 'LOS_normalization',
                         'draw_poission', 'log_mass_sheet_min', 'log_mass_sheet_max', 'kappa_scale']

        for key in required_keys:

            if key == 'LOS_normalization':

                if key in args.keys():
                    args_mfunc['LOS_normalization'] = args[key]
                else:
                    args_mfunc['LOS_normalization'] = 1
                continue

            if key == 'log_mlow_los':

                if key in args.keys():
                    args_mfunc['log_mlow'] = args[key]
                else:
                    args_mfunc['log_mlow'] = args['log_mlow']
                continue

            if key == 'log_mhigh_los':

                if key in args.keys():
                    args_mfunc['log_mhigh'] = args[key]
                else:
                    args_mfunc['log_mhigh'] = args['log_mhigh']
                continue

            try:

                args_mfunc[key] = args[key]

            except:
                if key == 'zmin':
                    args_mfunc['zmin'] = lenscone_default.default_zstart
                elif key == 'zmax':
                    args_mfunc['zmax'] = lens_mass_function.geometry._zsource - lenscone_default.default_zstart
                else:
                    raise Exception('Required keyword argument '+str(key) +' not specified.')

        if args_mfunc['log_m_break'] == 0:
            args_mfunc['break_index'] = 0
            args_mfunc['c_scale'] = 0
            args_mfunc['c_power'] = 0
            args_mfunc['break_scale'] = 1

        else:

            try:
                args_mfunc['break_index'] = args['break_index']
                args_mfunc['c_scale'] = args['c_scale']
                args_mfunc['c_power'] = args['c_power']
                args_mfunc['break_scale'] = args['break_scale']

            except:
                raise ValueError('must specify a value for "break_index, c_scale, c_power" if log_m_break != 0 '
                                 '(because you are specifying a WDM scenario in which the concentration and mass function'
                                 'slope  of halos is affected')

        return args_mfunc
