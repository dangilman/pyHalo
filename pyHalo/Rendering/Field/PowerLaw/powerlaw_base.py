from pyHalo.Rendering.Field.base import LOSBase
from pyHalo.defaults import *
from pyHalo.Rendering.MassFunctions.PowerLaw.broken_powerlaw import BrokenPowerLaw
from copy import deepcopy
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, \
    integrate_power_law_analytic

class PowerLawBase(LOSBase):

    def render_masses(self, zi, delta_zi, aperture_radius=None):

        """

        :param zi: redshift at which to render masses
        :param delta_zi: thickness of the redshift slice
        :param aperture_radius: the radius of the circular aperture where halos are rendered
        For DOUBLE_CONE and CYLINDER geometries this defaults to None and the aperture radius is computed in the geometry
        class automatically, resulting in a double-cone or cylinder rendering volume
        :return: halo masses at the desired redshift in units Msun
        """

        volume_element_comoving = self.geometry.volume_element_comoving(zi, delta_zi, aperture_radius)

        plaw_index = self.halo_mass_function.plaw_index_z(zi) + self.rendering_args['delta_power_law_index']

        norm = self.normalization(zi, delta_zi, self._zlens, self.halo_mass_function, self.rendering_args['host_m200'],
                                  volume_element_comoving, self.rendering_args['LOS_normalization'], plaw_index,
                                  self.rendering_args['m_pivot'])

        args = deepcopy(self.rendering_args)

        args.update({'normalization': norm})
        args.update({'power_law_index': plaw_index})

        mfunc = BrokenPowerLaw(args['log_mlow'], args['log_mhigh'], plaw_index, args['draw_poisson'],
                               norm, args['log_m_break'], args['break_index'], args['break_scale'])

        m = mfunc.draw()

        return m

    def _convergence_at_z(self, z, delta_z, log_sheet_min,
                             log_sheet_max, kappa_scale):

        volume_element_comoving = self.geometry.volume_element_comoving(z, delta_z, None)

        plaw_index = self.halo_mass_function.plaw_index_z(z) + self.rendering_args['delta_power_law_index']

        norm = self.normalization(z, delta_z, self._zlens, self.halo_mass_function, self.rendering_args['host_m200'],
                                  volume_element_comoving, self.rendering_args['LOS_normalization'], plaw_index,
                                  self.rendering_args['m_pivot'])

        if self.rendering_args['log_m_break'] is None or self.rendering_args['log_m_break'] == 0\
            or self.rendering_args['break_index'] is None or self.rendering_args['break_scale'] is None:
            use_analytic = True
        else:
            use_analytic = False

        m_low = 10 ** log_sheet_min
        m_high = 10 ** log_sheet_max

        if use_analytic:
            mtheory = integrate_power_law_analytic(norm, m_low, m_high, 1, plaw_index)
        else:
            mtheory = integrate_power_law_quad(norm, m_low, m_high, self.rendering_args['log_m_break'], 1,
                                            plaw_index, self.rendering_args['break_index'],
                                            self.rendering_args['break_scale'])

        area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, z)
        sigma_crit_mass = self.lens_cosmo.sigma_crit_mass(z, area)

        return kappa_scale * mtheory / sigma_crit_mass

    def negative_kappa_sheets_theory(self):

        kwargs_mass_sheets = self.keys_convergence_sheets

        log_mass_sheet_correction_min, log_mass_sheet_correction_max = \
            kwargs_mass_sheets['log_mass_sheet_min'], kwargs_mass_sheets['log_mass_sheet_max']

        kappa_scale = kwargs_mass_sheets['kappa_scale']

        lens_plane_redshifts = self.lens_plane_redshifts[0::2]
        delta_zs = 2*self.delta_zs[0::2]

        kwargs_out = []
        profile_names_out = []
        redshifts = []

        for z, delta_z in zip(lens_plane_redshifts, delta_zs):

            if z < kwargs_mass_sheets['zmin']:
                continue
            if z > kwargs_mass_sheets['zmax']:
                continue

            kappa = self._convergence_at_z(z, delta_z, log_mass_sheet_correction_min, log_mass_sheet_correction_max,
                                           kappa_scale)

            if kappa > 0:

                kwargs_out.append({'kappa_ext': -kappa})
                profile_names_out += ['CONVERGENCE']
                redshifts.append(z)

        return kwargs_out, profile_names_out, redshifts

    def normalization(self, z, delta_z, zlens, lensing_mass_function_class, host_mass, volume_element_comoving,
                      scale, plaw_index, m_pivot):

        norm_dV = scale * lensing_mass_function_class.norm_at_z_density(z, plaw_index, m_pivot)

        # boost will == 1 away from the lens redshift
        boost = self.two_halo_boost(z, delta_z, host_mass, zlens, lensing_mass_function_class)
        # add correlated structure at the lens redshift? Since the redshift slices are >> virial radius
        # we'll still add line of sight halos here....
        norm_dV *= boost

        return norm_dV * volume_element_comoving

    @property
    def keys_convergence_sheets(self):

        args_convergence_sheets = {}
        required_keys = ['log_mass_sheet_min', 'log_mass_sheet_max', 'kappa_scale', 'zmin', 'zmax',
                         'delta_power_law_index']

        raise_error = False
        missing_list = []
        for key in required_keys:
            if key not in self.rendering_args.keys():
                raise_error = True
                missing_list.append(key)
            else:
                args_convergence_sheets[key] = self.rendering_args[key]

        if raise_error:
            text = 'When specifying mass function type POWER_LAW and rendering line of sight halos, must provide all ' \
                   'required keyword arguments. The following need to be specified: '
            for key in missing_list:
                text += str(key) + '\n'
            raise Exception(text)

        return args_convergence_sheets

    @staticmethod
    def keyword_parse(args, lens_mass_function):

        args_mfunc = {}
        required_keys = ['zmin', 'zmax', 'log_m_break', 'log_mlow',
                         'log_mhigh', 'host_m200', 'LOS_normalization',
                         'draw_poisson', 'log_mass_sheet_min', 'log_mass_sheet_max', 'kappa_scale',
                         'break_index', 'break_scale', 'delta_power_law_index',
                         'm_pivot']

        for key in required_keys:

            try:

                args_mfunc[key] = args[key]

            except:
                if key == 'zmin':
                    args_mfunc['zmin'] = lenscone_default.default_zstart
                elif key == 'zmax':
                    args_mfunc['zmax'] = lens_mass_function.geometry._zsource - lenscone_default.default_zstart
                else:
                    raise Exception('Required keyword argument '+str(key) +' not specified.')

        if args_mfunc['log_m_break'] is None:
            args_mfunc['break_index'] = None
            args_mfunc['c_scale'] = None
            args_mfunc['c_power'] = None
            args_mfunc['break_scale'] = None

        return args_mfunc
