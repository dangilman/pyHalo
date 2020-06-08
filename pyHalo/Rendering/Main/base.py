from pyHalo.Rendering.MassFunctions.PowerLaw.broken_powerlaw import BrokenPowerLaw
from pyHalo.Spatial.nfw import NFW3D
from pyHalo.Spatial.uniform import Uniform
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Spatial.keywords import subhalo_spatial_NFW, subhalo_spatial_uniform
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, \
    integrate_power_law_analytic
from pyHalo.Rendering.Main.SHMF_normalizations import *

from pyHalo.Rendering.render_base import RenderingBase

class MainLensBase(RenderingBase):

    def __init__(self, args, geometry, x_center_lens, y_center_lens):

        zlens, zsource = geometry._zlens, geometry._zsource

        kpc_per_arcsec_zlens = geometry._kpc_per_arcsec_zlens

        lenscosmo = LensCosmo(zlens, zsource, geometry._cosmo)

        self.rendering_args = self.keyword_parse(args, kpc_per_arcsec_zlens, zlens)

        if 'subhalo_spatial_distribution' not in args.keys():
            raise Exception('must specify a value for the subhalo_spatial_distribution keyword.'
                            ' Possibilities are UNIFORM OR HOST_NFW.')

        if args['subhalo_spatial_distribution'] == 'UNIFORM':
            raise Exception('UNIFORM subhalo distribution not yet implemented.')
            spatial_args = subhalo_spatial_uniform(args)
            spatial_args['geometry'] = geometry
            spatial_class = Uniform

        elif args['subhalo_spatial_distribution'] == 'HOST_NFW':
            spatial_args = subhalo_spatial_NFW(args, kpc_per_arcsec_zlens, zlens, lenscosmo)
            spatial_class = NFW3D

        else:
            raise Exception('subhalo_spatial_distribution '+str(args['subhalo_spatial_distribution'])+
                            ' not recognized. Possibilities are UNIFORM OR HOST_NFW.')

        self._mass_func_parameterization = BrokenPowerLaw(**self.rendering_args)

        self.spatial_parameterization = spatial_class(**spatial_args)

        self._center_x, self._center_y = x_center_lens, y_center_lens

        super(MainLensBase, self).__init__(geometry)

    def negative_kappa_sheets_theory(self):

        kwargs_mass_sheets = self.keys_convergence_sheets

        if kwargs_mass_sheets['subtract_subhalo_mass_sheet'] is False:
            return [], [], []

        else:
            return self._negative_kappa_sheets_theory(kwargs_mass_sheets)

    def _negative_kappa_sheets_theory(self, kwargs_mass_sheets):

        log_mass_sheet_correction_min, log_mass_sheet_correction_max = \
            kwargs_mass_sheets['log_mass_sheet_min'], kwargs_mass_sheets['log_mass_sheet_max']

        kappa_scale = kwargs_mass_sheets['subhalo_mass_sheet_scale']

        m_low, m_high = 10 ** log_mass_sheet_correction_min, 10 ** log_mass_sheet_correction_max

        log_m_break = self.rendering_args['log_m_break']
        break_index = self.rendering_args['break_index']
        break_scale = self.rendering_args['break_scale']

        moment = 1

        if log_m_break == 0 or log_m_break / log_mass_sheet_correction_min < 0.01:
            use_analytic = True
        else:
            use_analytic = False

        norm = self.rendering_args['normalization']
        plaw_index = self.rendering_args['power_law_index']

        if use_analytic:
            mass_in_subhalos = integrate_power_law_analytic(norm, m_low, m_high, moment, plaw_index)
        else:
            mass_in_subhalos = integrate_power_law_quad(norm, m_low, m_high, log_m_break, moment,
                                            plaw_index, break_index, break_scale)

        
        if kwargs_mass_sheets['subhalo_convergence_correction_profile'] == 'UNIFORM':

            kappa = mass_in_subhalos / self.lens_cosmo.sigma_crit_mass(self.geometry._zlens, self.geometry)

            negative_kappa = -1 * kappa_scale * kappa

            kwargs_out = [{'kappa_ext': negative_kappa}]
            profile_name_out = ['CONVERGENCE']
            redshifts_out = [self.geometry._zlens]

        elif kwargs_mass_sheets['subhalo_convergence_correction_profile'] == 'NFW':

            if 'parent_m200' not in self.rendering_args.keys():
                raise Exception('must specify host halo mass when using NFW convergence sheet correction for subhalos')

            host_m200 = self.rendering_args['parent_m200']

            Rs_angle, _ = self.lens_cosmo.nfw_physical2angle_fromM(host_m200,
                                                                          self.geometry._zlens)

            c = self.lens_cosmo.NFW_concentration(host_m200, self.lens_cosmo.z_lens)
            _, rs_mpc, _ = self.lens_cosmo._nfwParam_physical_Mpc(host_m200, c, self.lens_cosmo.z_lens)

            x = self.geometry.cone_opening_angle / Rs_angle / 2

            eps_crit = self.lens_cosmo.epsilon_crit
            D_d = self.lens_cosmo.cosmo.D_A_z(self.geometry._zlens)
            denom = 4 * np.pi * rs_mpc ** 3 * (np.log(x/2) + self._nfw_F(x))
            rho0 = mass_in_subhalos/denom # solar mass per Mpc^3

            if isinstance(kwargs_mass_sheets['r_tidal'], str):
                r_core = Rs_angle * float(kwargs_mass_sheets['r_tidal'][0:-2])
            else:
                r_core = Rs_angle * kwargs_mass_sheets['r_tidal']

            theta_Rs = rho0 * (4 * rs_mpc ** 2 * (1 + np.log(1. / 2.)))
            theta_Rs *= 1 / (D_d * eps_crit * self.lens_cosmo.cosmo.arcsec)
            kwargs_out = [{'alpha_Rs': - kappa_scale * theta_Rs, 'Rs': Rs_angle,
                           'center_x': 0., 'center_y': 0., 'r_core': r_core}]

            profile_name_out = ['CNFW']
            redshifts_out = [self.geometry._zlens]

        else:
            raise Exception('mass sheet correction type '+
                            str(kwargs_mass_sheets['mass_sheet_correction_type']) + ' not recognized.')

        return kwargs_out, profile_name_out, redshifts_out

    def _nfw_F(self, x):

        if x == 1:
            return 1
        elif x < 1:
            f = np.sqrt(1 - x ** 2)
            return np.arctanh(f)/f
        else:
            f = np.sqrt(x ** 2 - 1)
            return np.arctan(f)/f

    @property
    def keys_convergence_sheets(self):

        args_convergence_sheets = {}
        required_keys = ['log_mass_sheet_min', 'log_mass_sheet_max', 'subhalo_mass_sheet_scale',
                         'subtract_subhalo_mass_sheet', 'subhalo_convergence_correction_profile',
                         'r_tidal']

        for key in required_keys:
            if key not in self.rendering_args.keys():
                raise Exception('When specifying mass function type POWER_LAW and rendering subhalos, must provide '
                                'key word arguments log_mass_sheet_min, log_mass_sheet_max, subtract_subhalo_mass_sheet, '
                                'and subhalo_mass_sheet_scale. These key words specify the halo mass '
                                'range used to add the convergence correction.')

            args_convergence_sheets[key] = self.rendering_args[key]

        return args_convergence_sheets

    @staticmethod
    def keyword_parse(args, kpc_per_arcsec_zlens, zlens):

        args_mfunc = {}

        required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_m_break',
                         'break_index', 'break_scale', 'log_mass_sheet_min', 'log_mass_sheet_max',
                         'subtract_subhalo_mass_sheet', 'subhalo_mass_sheet_scale', 'draw_poisson',
                         'subhalo_convergence_correction_profile', 'parent_m200', 'r_tidal']

        for key in required_keys:
            try:
                args_mfunc[key] = args[key]
            except:
                raise ValueError('must specify a value for ' + key)

        if 'sigma_sub' in args.keys():

            args_mfunc['normalization'] = norm_AO_from_sigmasub(args['sigma_sub'], args['parent_m200'],
                                                                zlens,
                                                                kpc_per_arcsec_zlens,
                                                                args['cone_opening_angle'],
                                                                args['power_law_index'])


        elif 'norm_kpc2' in args.keys():

            args_mfunc['normalization'] = norm_A0_from_a0area(args['norm_kpc2'],
                                                              zlens,
                                                              args['cone_opening_angle'],
                                                              args_mfunc['power_law_index'])

        elif 'norm_arcsec2' in args.keys():

            args_mfunc['normalization'] = norm_constant_per_squarearcsec(args['norm_arcsec2'],
                                                                         kpc_per_arcsec_zlens,
                                                                         args['cone_opening_angle'],
                                                                         args_mfunc['power_law_index'])

        elif 'f_sub' in args.keys() or 'log_f_sub' in args.keys():

            if 'log_f_sub' in args.keys():
                args['f_sub'] = 10 ** args['log_f_sub']

            a0_area_parent_halo = convert_fsub_to_norm(
                args['f_sub'], args['parent_m200'], zlens, args['R_ein_main'], args['cone_opening_angle'],
                zlens,
                args_mfunc['power_law_index'], 10 ** args_mfunc['log_mlow'],
                                               10 ** args_mfunc['log_mhigh'], mpivot=10 ** 8)

            args_mfunc['normalization'] = norm_A0_from_a0area(a0_area_parent_halo,
                                                              kpc_per_arcsec_zlens,
                                                              args['cone_opening_angle'],
                                                              args_mfunc['power_law_index'], m_pivot=10 ** 8)


        else:
            routines = 'sigma_sub: amplitude of differential mass function at 10^8 solar masses (d^2N / dmdA) in units [kpc^-2 M_sun^-1];\n' \
                       'automatically accounts for evolution of projected number density with halo mass and redshift (see Gilman et al. 2020)\n\n' \
                       'norm_kpc2: same as sigma_sub, but does not automatically account for evolution with halo mass and redshift\n\n' \
                       'norm_arcsec2: same as norm_kpc2, but in units (d^2N / dmdA) in units [arcsec^-2 M_sun^-1]\n\n' \
                       'f_sub or log_f_sub: projected mass fraction in substructure within the radius 0.5*cone_opening_angle'

            raise Exception('Must specify normalization of the subhalo '
                            'mass function. Recognized normalization routines are: \n' + routines)

        return args_mfunc

