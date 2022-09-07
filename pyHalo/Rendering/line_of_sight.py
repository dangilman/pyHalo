import numpy as np
from copy import deepcopy
from pyHalo.Rendering.MassFunctions.power_law import GeneralPowerLaw
from pyHalo.Rendering.MassFunctions.power_law_MixDM import GeneralPowerLawMixDM
from pyHalo.Rendering.MassFunctions.delta import DeltaFunction
from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, integrate_power_law_analytic, integrate_power_law_quad_MixDM
from pyHalo.Rendering.rendering_class_base import RenderingClassBase

class LineOfSightNoSheet(RenderingClassBase):
    """
    This class generates line-of-sight halos, or more precisely objects between the observer and the source that are
    not associated with the host dark matter halo around the main deflector.
    """

    def __init__(self, keywords_master, halo_mass_function, geometry, lens_cosmo,
                 lens_plane_redshifts, delta_z_list):

        """

        :param keywords_master: a dictionary of keyword arguments to be passed to each model class
        :param lens_cosmo: an instance of LensCosmo (see Halos.lens_cosmo)
        :param geometry: an instance of Geometry (see Cosmology.geometry)
        :param halo_mass_function: an instance of LensingMassFunction (see Cosmology.lensing_mass_function)
        :param lens_plane_redshifts: a list of redshifts at which to render halos
        :param delta_z_list: a list of redshift increments between each lens plane (should be the same length as
        lens_plane_redshifts)
        """

        self._rendering_kwargs = self.keyword_parse_render(keywords_master)
        self._keywords_master = keywords_master

        self.lens_cosmo = lens_cosmo

        self.spatial_distribution_model = LensConeUniform(keywords_master['cone_opening_angle'],
                                                          geometry)

        self.halo_mass_function = halo_mass_function
        self.geometry = geometry
        self._lens_plane_redshifts = lens_plane_redshifts
        self._delta_z_list = delta_z_list
        super(LineOfSightNoSheet, self).__init__()

    def render(self):

        """
        Generates halo masses and positions for objects along the line of sight
        (except for halos from the two-halo contribution)
        :return: mass (in Msun), x (arcsec), y (arcsec), r3d (kpc), redshift
        """

        masses = np.array([])
        x = np.array([])
        y = np.array([])
        redshifts = np.array([])

        for z, dz in zip(self._lens_plane_redshifts, self._delta_z_list):
            m = self.render_masses_at_z(z, dz)
            nhalos = len(m)
            _x, _y = self.render_positions_at_z(z, nhalos)
            _z = np.array([z] * len(_x))
            masses = np.append(masses, m)
            x = np.append(x, _x)
            y = np.append(y, _y)
            redshifts = np.append(redshifts, _z)

        subhalo_flag = [False] * len(masses)
        r3d = np.array([None] * len(masses))

        return masses, x, y, r3d, redshifts, subhalo_flag

    def render_positions_at_z(self, z, nhalos):

        """
        :param z: redshift
        :param nhalos: number of halos or objects to generate
        :return: the x, y coordinate of objects in arcsec, and a 3 dimensional coordinate in kpc
        The 3d coordinate only has a clear physical interpretation for subhalos, and is used to compute truncation raddi.
        For line of sight halos it is set to None.
        """

        x_kpc, y_kpc = self.spatial_distribution_model.draw(nhalos, z)

        if len(x_kpc) > 0:
            kpc_per_asec = self.geometry.kpc_per_arcsec(z)
            x_arcsec = x_kpc * kpc_per_asec ** -1
            y_arcsec = y_kpc * kpc_per_asec ** -1
            return x_arcsec, y_arcsec

        else:
            return np.array([]), np.array([])

    def render_masses_at_z(self, z, delta_z):

        """
        :param z: redshift at which to render masses
        :param delta_z: thickness of the redshift slice
        :return: halo masses at the desired redshift in units Msun
        """
        print('render masses at z is called')

        if self._rendering_kwargs['mass_function_LOS_type'] == 'POWER_LAW':
            print('PL type los mass function called')

            norm, plaw_index = self._normalization_slope(z, delta_z)

            args = deepcopy(self._rendering_kwargs)

            args.update({'normalization': norm})
            args.update({'power_law_index': plaw_index})

            log_mlow, log_mhigh = self._redshift_dependent_mass_range(z, args['log_mlow'], args['log_mhigh'])

            if 'frac' in self._keywords_master:
                mfunc = GeneralPowerLawMixDM(log_mlow, log_mhigh, plaw_index, args['draw_poisson'],
                                        norm, args['log_mc'], args['a_wdm'], args['b_wdm'],
                                        args['c_wdm'],self._keywords_master['frac'])
            else:
                mfunc = GeneralPowerLaw(log_mlow, log_mhigh, plaw_index, args['draw_poisson'],
                                    norm, args['log_mc'], args['a_wdm'], args['b_wdm'],
                                    args['c_wdm'])

        elif self._rendering_kwargs['mass_function_LOS_type'] == 'DELTA':

            volume = self.geometry.volume_element_comoving(z, delta_z)
            rho = self._rendering_kwargs['mass_fraction'] * self.lens_cosmo.cosmo.rho_dark_matter_crit
            mfunc = DeltaFunction(10 ** self._rendering_kwargs['logM'], volume, rho)

        else:
            raise Exception(
                'mass function type ' + str(self._rendering_kwargs['mass_function_type']) + ' not recognized')

        m = mfunc.draw()

        return m

    @staticmethod
    def keyword_parse_render(keywords_master):

        args_mfunc = {}

        required_keys = ['zmin', 'zmax',  'host_m200', 'LOS_normalization',
                         'draw_poisson', 'log_mass_sheet_min', 'log_mass_sheet_max', 'kappa_scale',
                         'mass_function_LOS_type']

        required_keys_power_law = ['a_wdm', 'b_wdm', 'c_wdm',
                                   'delta_power_law_index', 'm_pivot', 'log_mc', 'log_mlow', 'log_mhigh']
        required_keyes_delta = ['logM', 'mass_fraction']

        for key in required_keys:

            if key not in keywords_master:
                raise Exception('Required keyword argument ' + str(key) + ' not specified.')
            else:
                args_mfunc[key] = keywords_master[key]

        if keywords_master['mass_function_LOS_type'] == 'POWER_LAW':

            if keywords_master['log_mc'] is None:
                args_mfunc['a_wdm'] = None
                args_mfunc['b_wdm'] = None
                args_mfunc['c_wdm'] = None
                args_mfunc['c_scale'] = None
                args_mfunc['c_power'] = None
                args_mfunc['a_mc'] = None
                args_mfunc['b_mc'] = None
                args_mfunc['log_mc'] = None

            for key in required_keys_power_law:
                if key not in keywords_master:
                    raise Exception('Required keyword argument ' + str(key) + ' not specified.')
                else:
                    args_mfunc[key] = keywords_master[key]

        elif keywords_master['mass_function_LOS_type'] == 'DELTA':
            for key in required_keyes_delta:
                if key not in keywords_master:
                    raise Exception('Required keyword argument ' + str(key) + ' not specified.')
                else:
                    args_mfunc[key] = keywords_master[key]

        return args_mfunc

    def _normalization_slope(self, z, delta_z):

        los_norm = self._redshift_dependent_normalization(z, self._rendering_kwargs['LOS_normalization'])
        volume_element_comoving = self.geometry.volume_element_comoving(z, delta_z)
        plaw_index = self.halo_mass_function.plaw_index_z(z) + self._rendering_kwargs['delta_power_law_index']
        norm_dv = self.halo_mass_function.norm_at_z_density(z, plaw_index, self._rendering_kwargs['m_pivot'])
        norm = los_norm * norm_dv * volume_element_comoving
        return norm, plaw_index

    @staticmethod
    def keys_convergence_sheets(keywords_master):
        return {}

    def convergence_sheet_correction(self, kwargs_mass_sheets=None):

        return [{}], [], []


class LineOfSight(LineOfSightNoSheet):

    """
    This class generates line-of-sight halos, or more precisely objects between the observer and the source that are
    not associated with the host dark matter halo around the main deflector.
    """

    def __init__(self, keywords_master, halo_mass_function, geometry, lens_cosmo,
                 lens_plane_redshifts, delta_z_list):

        """

        :param keywords_master: a dictionary of keyword arguments to be passed to each model class
        :param lens_cosmo: an instance of LensCosmo (see Halos.lens_cosmo)
        :param geometry: an instance of Geometry (see Cosmology.geometry)
        :param halo_mass_function: an instance of LensingMassFunction (see Cosmology.lensing_mass_function)
        :param lens_plane_redshifts: a list of redshifts at which to render halos
        :param delta_z_list: a list of redshift increments between each lens plane (should be the same length as
        lens_plane_redshifts)
        """
        self._convergence_sheet_kwargs = self.keys_convergence_sheets(keywords_master)
        super(LineOfSight, self).__init__(keywords_master, halo_mass_function, geometry, lens_cosmo,
                 lens_plane_redshifts, delta_z_list)

    @staticmethod
    def keys_convergence_sheets(keywords_master):

        args_convergence_sheets = {}
        required_keys = ['log_mass_sheet_min', 'log_mass_sheet_max', 'kappa_scale', 'zmin', 'zmax',
                         'delta_power_law_index']

        raise_error = False
        missing_list = []
        for key in required_keys:
            if key not in keywords_master.keys():
                raise_error = True
                missing_list.append(key)
            else:
                args_convergence_sheets[key] = keywords_master[key]

        if raise_error:
            text = 'When specifying mass function type POWER_LAW and rendering line of sight halos, must provide all ' \
                   'required keyword arguments. The following need to be specified: '
            for key in missing_list:
                text += str(key) + '\n'
            raise Exception(text)

        return args_convergence_sheets

    def convergence_sheet_correction(self, kwargs_mass_sheets=None):

        """
        this routine applies the negative convergence sheet correction for lens planes along the line of sight
        :param kwargs_mass_sheets: keyword arguments that overwrite whatever the default settings for the mass sheet
        sheet are - leave it as None for most applications
        :return:
        """

        kw_mass_sheets = self._convergence_sheet_kwargs

        if kwargs_mass_sheets is not None:
            kw_mass_sheets.update(kwargs_mass_sheets)

        log_mass_sheet_correction_min, log_mass_sheet_correction_max = \
            kw_mass_sheets['log_mass_sheet_min'], kw_mass_sheets['log_mass_sheet_max']

        kappa_scale = kw_mass_sheets['kappa_scale']

        lens_plane_redshifts = self._lens_plane_redshifts[0::2]
        delta_zs = 2 * self._delta_z_list[0::2]

        kwargs_out = []
        profile_names_out = []
        redshifts = []

        for z, delta_z in zip(lens_plane_redshifts, delta_zs):

            if z < kw_mass_sheets['zmin']:
                continue
            if z > kw_mass_sheets['zmax']:
                continue

            log_mass_sheet_correction_min, log_mass_sheet_correction_max = self._redshift_dependent_mass_range(
                z, log_mass_sheet_correction_min, log_mass_sheet_correction_max
            )
            kappa = self._convergence_at_z(z, delta_z, log_mass_sheet_correction_min, log_mass_sheet_correction_max,
                                           kappa_scale)

            if kappa > 0:

                kwargs_out.append({'kappa': -kappa})
                profile_names_out += ['CONVERGENCE']
                redshifts.append(z)

        return kwargs_out, profile_names_out, redshifts

    def _convergence_at_z(self, z, delta_z, log_sheet_min,
                             log_sheet_max, kappa_scale):

        print('convergence at z is called')

        norm, plaw_index = self._normalization_slope(z, delta_z)

        m_low = 10 ** log_sheet_min
        m_high = 10 ** log_sheet_max

        if 'log_mc' in self._rendering_kwargs:
            if 'frac' in self._keywords_master:
                print('using MixDM suppression, los')
                mtheory = integrate_power_law_quad_MixDM(norm, m_low, m_high, self._rendering_kwargs['log_mc'], 1,
                                               plaw_index, self._rendering_kwargs['a_wdm'],
                                               self._rendering_kwargs['b_wdm'],
                                               self._rendering_kwargs['c_wdm'],self._rendering_kwargs['frac'])
            else:
                print('using wdm suppression, los')
                mtheory = integrate_power_law_quad(norm, m_low, m_high, self._rendering_kwargs['log_mc'], 1,
                                               plaw_index, self._rendering_kwargs['a_wdm'],
                                               self._rendering_kwargs['b_wdm'],
                                               self._rendering_kwargs['c_wdm'])
        else:
            mtheory = integrate_power_law_analytic(norm, m_low, m_high, 1, plaw_index)

        area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, z)
        sigma_crit_mass = self.lens_cosmo.sigma_crit_mass(z, area)

        return kappa_scale * mtheory / sigma_crit_mass
