from pyHalo.Rendering.MassFunctions.PowerLaw.broken_powerlaw import BrokenPowerLaw
from pyHalo.Rendering.MassFunctions.PowerLaw.piecewise import PiecewisePowerLaw
from pyHalo.Spatial.nfw_core import NFW3DCoreRejectionSampling, CoreNFW3DFast, UniformNFW
from pyHalo.Spatial.uniform import Uniform
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Spatial.keywords import subhalo_spatial_NFW, subhalo_spatial_uniform
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, \
    integrate_power_law_analytic
from pyHalo.Rendering.Main.SHMF_normalizations import *

from copy import deepcopy

from pyHalo.Rendering.render_base import RenderingBase

class MainLensBase(RenderingBase):

    type = 'main_lens_plane'

    def __init__(self, args, geometry, x_center_lens, y_center_lens):

        zlens, zsource = geometry._zlens, geometry._zsource

        kpc_per_arcsec_zlens = geometry._kpc_per_arcsec_zlens

        lenscosmo = LensCosmo(zlens, zsource, geometry._cosmo)

        if 'subhalo_spatial_distribution' not in args.keys():
            raise Exception('must specify a value for the subhalo_spatial_distribution keyword.'
                            ' Possibilities are UNIFORM OR HOST_NFW.')

        if args['subhalo_spatial_distribution'] == 'UNIFORM':
            if args['subtract_subhalo_mass_sheet'] and args['subhalo_convergence_correction_profile']=='NFW':
                raise Exception('should not implement an NFW covergence correction with '
                                'a unfiormly-distributed population of subhalos')
            #raise Exception('UNIFORM subhalo distribution not yet implemented.')
            spatial_args = subhalo_spatial_uniform(args)
            spatial_args['geometry'] = geometry
            spatial_class = Uniform

        elif args['subhalo_spatial_distribution'] == 'HOST_NFW':
            spatial_args = subhalo_spatial_NFW(args, kpc_per_arcsec_zlens, zlens, lenscosmo)
            spatial_class = UniformNFW

        else:
            raise Exception('subhalo_spatial_distribution '+str(args['subhalo_spatial_distribution'])+
                            ' not recognized. Possibilities are UNIFORM OR HOST_NFW.')

        self._mass_func_parameterization, self.rendering_args = self.keyword_parse(args, kpc_per_arcsec_zlens, zlens)

        self._spatial_args = spatial_args

        self.spatial_parameterization = spatial_class(**spatial_args)

        self._center_x, self._center_y = x_center_lens, y_center_lens

        self.convergence_correction_centroid_x = 0.
        self.convergence_correction_centroid_y = 0.

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

        mass_in_subhalos = self._mass_func_parameterization.theory_mass(m_low, m_high)

        if kwargs_mass_sheets['subhalo_convergence_correction_profile'] == 'UNIFORM':

            kappa = mass_in_subhalos / self.lens_cosmo.sigma_crit_mass(self.geometry._zlens, self.geometry)

            negative_kappa = -1 * kappa_scale * kappa

            kwargs_out = [{'kappa_ext': negative_kappa}]
            profile_name_out = ['CONVERGENCE']
            redshifts_out = [self.geometry._zlens]

        elif kwargs_mass_sheets['subhalo_convergence_correction_profile'] == 'NFW':

            if 'parent_m200' not in self.rendering_args.keys():
                raise Exception('must specify host halo mass when using NFW convergence sheet correction for subhalos')

            #host_m200 = self.rendering_args['parent_m200']

            Rs_host_kpc = self._spatial_args['Rs']
            rs_mpc = Rs_host_kpc / 1000
            r_tidal_host_kpc = self._spatial_args['r_core_parent']

            Rs_angle = Rs_host_kpc / self.geometry._kpc_per_arcsec_zlens
            r_core_angle = r_tidal_host_kpc / self.geometry._kpc_per_arcsec_zlens

            x = self.geometry.cone_opening_angle / Rs_angle / 2

            eps_crit = self.lens_cosmo.epsilon_crit
            D_d = self.lens_cosmo.cosmo.D_A_z(self.geometry._zlens)
            denom = 4 * np.pi * rs_mpc ** 3 * (np.log(x/2) + self._nfw_F(x))
            rho0 = mass_in_subhalos/denom # solar mass per Mpc^3

            theta_Rs = rho0 * (4 * rs_mpc ** 2 * (1 + np.log(1. / 2.)))
            theta_Rs *= 1 / (D_d * eps_crit * self.lens_cosmo.cosmo.arcsec)

            kwargs_out = [{'alpha_Rs': - kappa_scale * theta_Rs, 'Rs': Rs_angle,
                           'center_x': kwargs_mass_sheets['nfw_kappa_centroid'][0],
                           'center_y': kwargs_mass_sheets['nfw_kappa_centroid'][1], 'r_core': r_core_angle}]

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
                         'r_tidal', 'nfw_kappa_centroid', 'delta_power_law_index']

        raise_error = False
        for key in required_keys:
            missing_list = []
            if key not in self.rendering_args.keys():
                raise_error = True
                missing_list.append(key)
            else:
                args_convergence_sheets[key] = self.rendering_args[key]

        if raise_error:
            text = 'When specifying mass function type POWER_LAW and rendering subhalos, must provide all ' \
                   'required keyword arguments. The following need to be specified: '
            for key in missing_list:
                text += str(key) + '\n'
            raise Exception(text)

        return args_convergence_sheets

    def keyword_parse(self, args, kpc_per_arcsec_zlens, zlens):

        possible_normalization_kwargs = ['sigma_sub', 'norm_kpc2',
                                         'norm_arcsec2', 'f_sub', 'log_f_sub', 'log_sigma_sub']
        for kw in possible_normalization_kwargs:
            if kw in args.keys():
                break
        else:
            raise Exception('must specify SHMF normalization with one of: ', possible_normalization_kwargs)

        # build rescale list if flag is provided
        if 'build_SHMF_rescale_list' in args.keys() or 'SHMF_rescale_list' in args.keys():
            assert isinstance(args['log_mlow'], list)
            assert isinstance(args['log_mhigh'], list)
            assert len(args['log_mlow']) == len(args['log_mhigh'])

            if 'build_SHMF_rescale_list' in args.keys() and args['build_SHMF_rescale_list'] is True:
                N = len(args['log_mlow'])
                kw_base = 'rescale_'
                SHMF_rescale_list = []
                for k in range(0, N):
                    pname = kw_base + str(int(k+1))
                    assert pname in args.keys()
                    SHMF_rescale_list.append(args[pname])
                args['SHMF_rescale_list'] = SHMF_rescale_list

            assert len(args['SHMF_rescale_list']) == len(args['log_mlow'])

            func = []
            args_mfunc_list = []

            args_mfunc_return = deepcopy(args)

            for i, rescale in enumerate(args['SHMF_rescale_list']):
                assert args['log_mlow'][i] <= args['log_mhigh'][i]

                args_copy = deepcopy(args)
                args_copy['rescale_norm'] = rescale
                args_copy['log_mlow'] = args['log_mlow'][i]
                args_copy['log_mhigh'] = args['log_mhigh'][i]

                if isinstance(args['power_law_index'], list):
                    args_copy['power_law_index'] = args['power_law_index'][i] + args['delta_power_law_index']

                args_mfunc = self._keyword_parse(args_copy, kpc_per_arcsec_zlens, zlens)
                func.append(BrokenPowerLaw(**args_mfunc))
                args_mfunc_list.append(args_mfunc)

            parameterization = PiecewisePowerLaw(func)

        else:

            args['rescale_norm'] = 1.
            args_mfunc_return = self._keyword_parse(args, kpc_per_arcsec_zlens, zlens)
            parameterization = BrokenPowerLaw(**args_mfunc_return)

        return parameterization, args_mfunc_return

    @staticmethod
    def _keyword_parse(args, kpc_per_arcsec_zlens, zlens):

        args_mfunc = {}

        required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_m_break',
                         'break_index', 'break_scale', 'log_mass_sheet_min', 'log_mass_sheet_max',
                         'subtract_subhalo_mass_sheet', 'subhalo_mass_sheet_scale', 'draw_poisson',
                         'subhalo_convergence_correction_profile', 'parent_m200', 'r_tidal',
                         'nfw_kappa_centroid', 'delta_power_law_index', 'm_pivot']

        assert not isinstance(args['log_mlow'], list)
        assert not isinstance(args['log_mhigh'], list)
        assert not isinstance(args['power_law_index'], list)

        for key in required_keys:
            try:
                args_mfunc[key] = args[key]
            except:
                raise ValueError('must specify a value for ' + key)

        args_mfunc['power_law_index'] += args_mfunc['delta_power_law_index']

        if 'sigma_sub' in args.keys():

            args_mfunc['normalization'] = norm_AO_from_sigmasub(args['sigma_sub'], args['parent_m200'],
                                                                zlens,
                                                                kpc_per_arcsec_zlens,
                                                                args['cone_opening_angle'],
                                                                args_mfunc['power_law_index'],
                                                                args['m_pivot'])

        elif 'log_sigma_sub' in args.keys():

            sigma_sub = 10 ** args['log_sigma_sub']
            args_mfunc['normalization'] = norm_AO_from_sigmasub(sigma_sub, args['parent_m200'],
                                                                zlens,
                                                                kpc_per_arcsec_zlens,
                                                                args['cone_opening_angle'],
                                                                args_mfunc['power_law_index'],
                                                                args['m_pivot'])

        elif 'norm_kpc2' in args.keys():

            args_mfunc['normalization'] = norm_A0_from_a0area(args['norm_kpc2'],
                                                              zlens,
                                                              args['cone_opening_angle'],
                                                              args_mfunc['power_law_index'],
                                                              args['m_pivot'])

        elif 'norm_arcsec2' in args.keys():

            args_mfunc['normalization'] = norm_constant_per_squarearcsec(args['norm_arcsec2'],
                                                                         kpc_per_arcsec_zlens,
                                                                         args['cone_opening_angle'],
                                                                         args_mfunc['power_law_index'],
                                                                         args['m_pivot'])

        elif 'f_sub' in args.keys() or 'log_f_sub' in args.keys():

            if 'log_f_sub' in args.keys():
                args['f_sub'] = 10 ** args['log_f_sub']

            a0_area_parent_halo = convert_fsub_to_norm(
                args['f_sub'], args['parent_m200'], zlens, args['R_ein_main'], args['cone_opening_angle'],
                zlens,
                args_mfunc['power_law_index'], 10 ** args_mfunc['log_mlow'],
                                               10 ** args_mfunc['log_mhigh'], mpivot=args['m_pivot'])

            args_mfunc['normalization'] = norm_A0_from_a0area(a0_area_parent_halo,
                                                              kpc_per_arcsec_zlens,
                                                              args['cone_opening_angle'],
                                                              args_mfunc['power_law_index'], m_pivot=args['m_pivot'])

        else:

            routines = 'sigma_sub: amplitude of differential mass function at 10^8 solar masses (d^2N / dmdA) in units [kpc^-2 M_sun^-1];\n' \
                       'automatically accounts for evolution of projected number density with halo mass and redshift (see Gilman et al. 2020)\n\n' \
                       'norm_kpc2: same as sigma_sub, but does not automatically account for evolution with halo mass and redshift\n\n' \
                       'norm_arcsec2: same as norm_kpc2, but in units (d^2N / dmdA) in units [arcsec^-2 M_sun^-1]\n\n' \
                       'f_sub or log_f_sub: projected mass fraction in substructure within the radius 0.5*cone_opening_angle'

            raise Exception('Must specify normalization of the subhalo '
                            'mass function. Recognized normalization routines are: \n' + routines)

        args_mfunc['normalization'] *= args['rescale_norm']

        return args_mfunc

