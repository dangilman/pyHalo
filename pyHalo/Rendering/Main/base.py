from pyHalo.Rendering.MassFunctions.PowerLaw.broken_powerlaw import BrokenPowerLaw
from pyHalo.Spatial.nfw_core import ProjectedNFW
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Spatial.keywords import subhalo_spatial_NFW
from pyHalo.Rendering.Main.SHMF_normalizations import *
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, integrate_power_law_analytic
from pyHalo.Rendering.render_base import RenderingBase

class MainLensBase(RenderingBase):

    type = 'main_lens_plane'

    def __init__(self, args, geometry):

        zlens, zsource = geometry._zlens, geometry._zsource

        kpc_per_arcsec_zlens = geometry._kpc_per_arcsec_zlens

        lenscosmo = LensCosmo(zlens, zsource, geometry._cosmo)

        if 'subhalo_spatial_distribution' not in args.keys():
            raise Exception('must specify a value for the subhalo_spatial_distribution keyword.'
                            ' Currently only HOST_NFW is implemented.')

        elif args['subhalo_spatial_distribution'] == 'HOST_NFW':
            spatial_args = subhalo_spatial_NFW(args, kpc_per_arcsec_zlens, zlens, lenscosmo)
            spatial_class = ProjectedNFW

        else:
            raise Exception('subhalo_spatial_distribution '+str(args['subhalo_spatial_distribution'])+
                            ' not recognized. Try HOST_NFW.')

        self.rendering_args = self.keyword_parse(args, kpc_per_arcsec_zlens, zlens)
        power_law_index = self.rendering_args['power_law_index'] + self.rendering_args['delta_power_law_index']
        self._mass_func_parameterization = BrokenPowerLaw(self.rendering_args['log_mlow'],
                                                          self.rendering_args['log_mhigh'],
                                                          power_law_index,
                                                          self.rendering_args['draw_poisson'],
                                                          self.rendering_args['normalization'],
                                                          self.rendering_args['log_mc'],
                                                          self.rendering_args['a_wdm'],
                                                          self.rendering_args['b_wdm'],
                                                          self.rendering_args['c_wdm'])

        self._spatial_args = spatial_args

        self.spatial_parameterization = spatial_class(**spatial_args)

        super(MainLensBase, self).__init__(geometry)

    def negative_kappa_sheets_theory(self):

        kwargs_mass_sheets = self.keys_convergence_sheets

        log_mass_sheet_correction_min, log_mass_sheet_correction_max = \
            kwargs_mass_sheets['log_mass_sheet_min'], kwargs_mass_sheets['log_mass_sheet_max']

        kappa_scale = kwargs_mass_sheets['subhalo_mass_sheet_scale']

        m_low, m_high = 10 ** log_mass_sheet_correction_min, 10 ** log_mass_sheet_correction_max

        power_law_index = self.rendering_args['power_law_index'] + self.rendering_args['delta_power_law_index']

        if self.rendering_args['log_mc'] is not None:
            mass_in_subhalos = integrate_power_law_quad(self.rendering_args['normalization'],
                                                        m_low, m_high, self.rendering_args['log_mc'],
                                                        1, power_law_index, self.rendering_args['a_wdm'],
                                                        self.rendering_args['b_wdm'], self.rendering_args['c_wdm'])
        else:
            mass_in_subhalos = integrate_power_law_analytic(self.rendering_args['normalization'],
                                                        m_low, m_high, 1, power_law_index)

        if kwargs_mass_sheets['subhalo_convergence_correction_profile'] == 'UNIFORM':

            area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, self._zlens)
            kappa = mass_in_subhalos / self.lens_cosmo.sigma_crit_mass(self._zlens, area)

            negative_kappa = -1 * kappa_scale * kappa

            kwargs_out = [{'kappa_ext': negative_kappa}]
            profile_name_out = ['CONVERGENCE']
            redshifts_out = [self._zlens]

        elif kwargs_mass_sheets['subhalo_convergence_correction_profile'] == 'NFW':

            if 'host_m200' not in self.rendering_args.keys():
                raise Exception('must specify host halo mass when using NFW convergence sheet correction for subhalos')

            Rs_host_kpc = self._spatial_args['Rs']
            rs_mpc = Rs_host_kpc / 1000
            r_tidal_host_kpc = self._spatial_args['r_core_parent']

            Rs_angle = Rs_host_kpc / self.geometry._kpc_per_arcsec_zlens
            r_core_angle = r_tidal_host_kpc / self.geometry._kpc_per_arcsec_zlens

            x = self.geometry.cone_opening_angle / Rs_angle / 2

            eps_crit = self.lens_cosmo.get_sigma_crit_lensing(self._zlens, self._zsource)
            D_d = self.lens_cosmo.cosmo.D_A_z(self._zlens)
            denom = 4 * np.pi * rs_mpc ** 3 * (np.log(x/2) + self._nfw_F(x))
            rho0 = mass_in_subhalos/denom # solar mass per Mpc^3

            theta_Rs = rho0 * (4 * rs_mpc ** 2 * (1 + np.log(1. / 2.)))
            theta_Rs *= 1 / (D_d * eps_crit * self.lens_cosmo.cosmo.arcsec)

            kwargs_out = [{'alpha_Rs': - kappa_scale * theta_Rs, 'Rs': Rs_angle,
                           'center_x': 0.,
                           'center_y': 0., 'r_core': r_core_angle}]

            profile_name_out = ['CNFW']
            redshifts_out = [self._zlens]

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
                         'subhalo_convergence_correction_profile',
                         'r_tidal', 'delta_power_law_index']

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

    @staticmethod
    def keyword_parse(args, kpc_per_arcsec_zlens, zlens):

        args_mfunc = {}

        required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_mc', 'sigma_sub',
                         'a_wdm', 'b_wdm', 'c_wdm', 'log_mass_sheet_min', 'log_mass_sheet_max',
                         'subhalo_mass_sheet_scale', 'draw_poisson', 'host_m200',
                         'subhalo_convergence_correction_profile', 'r_tidal',
                         'delta_power_law_index', 'm_pivot']

        for key in required_keys:
            try:
                args_mfunc[key] = args[key]
            except:
                raise ValueError('must specify a value for ' + key)

        power_law_index = args_mfunc['power_law_index'] + args_mfunc['delta_power_law_index']
        args_mfunc['normalization'] = normalization_sigmasub(args['sigma_sub'], args['host_m200'],
                                                             zlens,
                                                             kpc_per_arcsec_zlens,
                                                             args['cone_opening_angle'],
                                                             power_law_index,
                                                             args['m_pivot'])


        return args_mfunc

