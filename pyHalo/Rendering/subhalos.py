import numpy as np
from pyHalo.Rendering.MassFunctions.power_law import GeneralPowerLaw
from pyHalo.Rendering.SpatialDistributions.nfw_core import ProjectedNFW
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_analytic, integrate_power_law_quad
from pyHalo.Rendering.rendering_class_base import RenderingClassBase

class Subhalos(RenderingClassBase):

    def __init__(self, keywords_master, geometry, lens_cosmo):

        self._zlens = lens_cosmo.z_lens
        self._z_source = lens_cosmo.z_source
        self.geometry = geometry
        self.lens_cosmo = lens_cosmo
        self._convergence_sheet_kwargs = self.keys_convergence_sheets(keywords_master)
        self._rendering_kwargs = self.keyword_parse_render(keywords_master)

        if 'subhalo_spatial_distribution' not in keywords_master.keys():
            raise Exception('must specify a value for the subhalo_spatial_distribution keyword.'
                            ' Currently only HOST_NFW is implemented.')

        elif keywords_master['subhalo_spatial_distribution'] == 'HOST_NFW':
            self.spatial_distribution_model = ProjectedNFW.from_keywords_master(keywords_master, lens_cosmo, geometry)

        else:
            raise Exception('subhalo_spatial_distribution ' + str(keywords_master['subhalo_spatial_distribution']) +
                            ' not recognized. Try HOST_NFW.')

        super(Subhalos, self).__init__()

    def render(self):

        m = self.render_masses_at_z()
        x, y, r3d = self.render_positions_at_z(len(m))
        z = np.array([self._zlens] * len(m))
        subhalo_flag = [True] * len(m)

        return m, x, y, r3d, z, subhalo_flag

    def render_positions_at_z(self, nhalos):

        x_kpc, y_kpc, r3d_kpc = self.spatial_distribution_model.draw(nhalos)

        x_arcsec = np.array(x_kpc) / self.geometry.kpc_per_arcsec_zlens
        y_arcsec = np.array(y_kpc) / self.geometry.kpc_per_arcsec_zlens

        return np.array(x_arcsec), np.array(y_arcsec), np.array(r3d_kpc)

    def render_masses_at_z(self):

        norm, slope = self._norm_slope()

        mfunc = GeneralPowerLaw(self._rendering_kwargs['log_mlow'],
                                                           self._rendering_kwargs['log_mhigh'],
                                                           slope,
                                                           self._rendering_kwargs['draw_poisson'],
                                                           norm,
                                                           self._rendering_kwargs['log_mc'],
                                                           self._rendering_kwargs['a_wdm'],
                                                           self._rendering_kwargs['b_wdm'],
                                                           self._rendering_kwargs['c_wdm'])
        m = mfunc.draw()

        return m

    @staticmethod
    def keys_convergence_sheets(keywords_master):

        args_convergence_sheets = {}
        required_keys = ['log_mass_sheet_min', 'log_mass_sheet_max', 'subhalo_mass_sheet_scale',
                         'subhalo_convergence_correction_profile',
                         'r_tidal', 'delta_power_law_index', 'delta_power_law_index_coupling']

        raise_error = False
        missing_list = []

        for key in required_keys:
            if key not in keywords_master.keys():
                raise_error = True
                missing_list.append(key)
            else:
                args_convergence_sheets[key] = keywords_master[key]

        if raise_error:
            text = 'When specifying mass function type POWER_LAW and rendering subhalos, must provide all ' \
                   'required keyword arguments. The following need to be specified:\n'
            for key in missing_list:
                text += str(key) + '\n'
            raise Exception(text)

        return args_convergence_sheets

    @staticmethod
    def keyword_parse_render(keywrds_master):

        args_mfunc = {}

        required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_mc', 'sigma_sub',
                         'a_wdm', 'b_wdm', 'c_wdm', 'log_mass_sheet_min', 'log_mass_sheet_max',
                         'subhalo_mass_sheet_scale', 'draw_poisson', 'host_m200',
                         'subhalo_convergence_correction_profile', 'r_tidal',
                         'delta_power_law_index', 'm_pivot', 'delta_power_law_index_coupling',
                         'cone_opening_angle']

        for key in required_keys:
            try:
                args_mfunc[key] = keywrds_master[key]
            except:
                raise ValueError('must specify a value for ' + key)

        if args_mfunc['log_mc'] is None:
            args_mfunc['a_wdm'] = None
            args_mfunc['b_wdm'] = None
            args_mfunc['c_wdm'] = None
            args_mfunc['c_scale'] = None
            args_mfunc['c_power'] = None

        return args_mfunc

    def convergence_sheet_correction(self):

        norm, slope = self._norm_slope()

        kwargs_mass_sheets = self._convergence_sheet_kwargs

        log_mass_sheet_correction_min, log_mass_sheet_correction_max = \
            kwargs_mass_sheets['log_mass_sheet_min'], kwargs_mass_sheets['log_mass_sheet_max']

        kappa_scale = kwargs_mass_sheets['subhalo_mass_sheet_scale']

        m_low, m_high = 10 ** log_mass_sheet_correction_min, 10 ** log_mass_sheet_correction_max

        delta_power_law_index = self._rendering_kwargs['delta_power_law_index_coupling'] * self._rendering_kwargs[
            'delta_power_law_index']
        power_law_index = self._rendering_kwargs['power_law_index'] + delta_power_law_index

        if self._rendering_kwargs['log_mc'] is not None:
            mass_in_subhalos = integrate_power_law_quad(norm,
                                                        m_low, m_high, self._rendering_kwargs['log_mc'],
                                                        1, power_law_index, self._rendering_kwargs['a_wdm'],
                                                        self._rendering_kwargs['b_wdm'], self._rendering_kwargs['c_wdm'])
        else:
            mass_in_subhalos = integrate_power_law_analytic(norm,
                                                        m_low, m_high, 1, power_law_index)

        if kwargs_mass_sheets['subhalo_convergence_correction_profile'] == 'UNIFORM':

            area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, self._zlens)
            kappa = mass_in_subhalos / self.lens_cosmo.sigma_crit_mass(self._zlens, area)

            negative_kappa = -1 * kappa_scale * kappa

            kwargs_out = [{'kappa_ext': negative_kappa}]
            profile_name_out = ['CONVERGENCE']
            redshifts_out = [self._zlens]

        elif kwargs_mass_sheets['subhalo_convergence_correction_profile'] == 'NFW':

            if 'host_m200' not in self._rendering_kwargs.keys():
                raise Exception('must specify host halo mass when using NFW convergence sheet correction for subhalos')

            Rs_host_kpc = self.spatial_distribution_model._rs_kpc
            rs_mpc = Rs_host_kpc / 1000
            r_tidal_host_kpc = self.spatial_distribution_model.xtidal * Rs_host_kpc

            Rs_angle = Rs_host_kpc / self.geometry.kpc_per_arcsec_zlens
            r_core_angle = r_tidal_host_kpc / self.geometry.kpc_per_arcsec_zlens

            x = self.geometry.cone_opening_angle / Rs_angle / 2

            eps_crit = self.lens_cosmo.get_sigma_crit_lensing(self._zlens, self._z_source)
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
            raise Exception('subhalo convergence correction profile '+
                            str(kwargs_mass_sheets['subhalo_convergence_correction_profile']) + ' not recognized.')

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

    def _norm_slope(self):

        delta_power_law_index = self._rendering_kwargs['delta_power_law_index'] * \
                                self._rendering_kwargs['delta_power_law_index_coupling']
        slope = self._rendering_kwargs['power_law_index'] + delta_power_law_index
        norm = normalization_sigmasub(self._rendering_kwargs['sigma_sub'], self._rendering_kwargs['host_m200'],
                                      self._zlens,
                                      self.geometry.kpc_per_arcsec_zlens,
                                      self._rendering_kwargs['cone_opening_angle'],
                                      slope,
                                      self._rendering_kwargs['m_pivot'])

        return norm, slope

def host_scaling_function(mhalo, z, k1 = 0.88, k2 = 1.7):

    logscaling = k1 * np.log10(mhalo / 10**13) + k2 * np.log10(z + 0.5)

    return 10 ** logscaling

def normalization_sigmasub(sigma_sub, host_m200, zlens, kpc_per_asec_zlens, cone_opening_angle, plaw_index, m_pivot):

    a0_per_kpc2 = sigma_sub * host_scaling_function(host_m200, zlens)

    R_kpc = kpc_per_asec_zlens * (0.5 * cone_opening_angle)

    area = np.pi * R_kpc ** 2

    m_pivot_factor = m_pivot ** -(plaw_index+1)

    return area * a0_per_kpc2 * m_pivot_factor
