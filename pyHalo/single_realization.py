import numpy as np

from pyHalo.Halos.halo import Halo
from pyHalo.defaults import *
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Massfunc.mainlens import norm_AO_from_sigmasub
from scipy.integrate import quad

from copy import deepcopy

def realization_at_z(realization,z):

    halos = realization.halos_at_z(z)

    return Realization.from_halos(halos, realization.halo_mass_function,
                                  realization._prof_params, realization._mass_sheet_correction)

class Realization(object):

    def __init__(self, masses, x, y, r2d, r3d, mdefs, z, subhalo_flag, halo_mass_function,
                 halos=None, other_params = {}, mass_sheet_correction=True):

        self._mass_sheet_correction = mass_sheet_correction
        self._subtract_theory_mass_sheets = True
        self._overwrite_mass_sheet = None

        self.halo_mass_function = halo_mass_function
        self.geometry = halo_mass_function.geometry
        self.lens_cosmo = LensCosmo(self.geometry._zlens, self.geometry._zsource,
                                    self.geometry._cosmo)
        self._lensing_functions = []
        self.halos = []
        self._loaded_models = {}

        self._prof_params = set_default_kwargs(other_params)

        self.m_break_scale = self._prof_params['log_m_break']
        self.break_index = self._prof_params['break_index']
        self._LOS_norm = self._prof_params['LOS_normalization']
        self.break_scale = self._prof_params['break_scale']

        self._logmlow = self._prof_params['log_mlow']
        self._logmhigh = self._prof_params['log_mhigh']

        if halos is None:

            for mi, xi, yi, r2di, r3di, mdefi, zi, sub_flag in zip(masses, x, y, r2d, r3d,
                           mdefs, z, subhalo_flag):

                self._add_halo(mi, xi, yi, r2di, r3di, mdefi, zi, sub_flag)

            if self._prof_params['include_subhalos']:
                raise Exception('subhalos of halos not yet implemented.')

        else:

            for halo in halos:
                self._add_halo(None, None, None, None, None, None, None, None, halo=halo)

        self._reset()

    @classmethod
    def from_halos(cls, halos, halo_mass_function, prof_params, msheet_correction):

        realization = Realization(None, None, None, None, None, None, None, None, halo_mass_function,
                           halos=halos, other_params=prof_params,
                           mass_sheet_correction=msheet_correction)
        return realization



    def add_halo(self, mass, x, y, r2d, r3d, mdef, z, sub_flag):

        new_real = Realization([mass], [x], [y], [r2d], [r3d], [mdef], [z], [sub_flag], self.halo_mass_function,
                               halos = None, other_params=self._prof_params,
                               mass_sheet_correction=self._mass_sheet_correction)

        realization = self.join(new_real)
        return realization

    def change_profile_params(self, new_args):

        new_params = deepcopy(self._prof_params)
        new_params.update(new_args)

        return Realization(self.masses, self.x, self.y, self.r2d, self.r3d, self.mdefs,
                           self.redshifts, self.subhalo_flags, self.halo_mass_function,
                           other_params=new_params, mass_sheet_correction=self._mass_sheet_correction)

    def change_mdef(self, new_mdef):

        new_halos = []
        for halo in self.halos:
            duplicate = deepcopy(halo)
            if duplicate.mdef == 'cNFWmod_trunc' and new_mdef == 'TNFW':

                duplicate._mass_def_arg = duplicate.profile_args[0:-1]

            else:
                raise Exception('combination '+duplicate.mdef + ' and '+
                                    new_mdef+' not recognized.')

            duplicate.mdef = new_mdef
            new_halos.append(duplicate)

        return Realization(None, None, None, None, None, None, None, None, self.halo_mass_function,
                           halos = new_halos, other_params= self._prof_params,
                           mass_sheet_correction = self._mass_sheet_correction)

    def _tags(self, halos=None):

        if halos is None:
            halos = self.halos
        tags = []

        for halo in halos:

            tags.append(halo._unique_tag)

        return tags

    def join(self, real):
        """

        :param real: another realization, possibly a filtered version of self
        :return: a new realization with all unique halos from self and real
        """
        halos = []

        tags = self._tags(self.halos)
        real_tags = self._tags(real.halos)
        if len(tags) >= len(real_tags):
            long, short = tags, real_tags
            halos_long, halos_short = self.halos, real.halos
        else:
            long, short = real_tags, tags
            halos_long, halos_short = real.halos, self.halos

        for halo in halos_short:
            halos.append(halo)

        for i, tag in enumerate(long):

            if tag not in short:
                halos.append(halos_long[i])

        return Realization(None, None, None, None, None, None, None, None, self.halo_mass_function,
                           halos = halos, other_params= self._prof_params,
                           mass_sheet_correction = self._mass_sheet_correction)

    def _reset(self):

        self.x = []
        self.y = []
        self.masses = []
        self.redshifts = []
        self.r2d = []
        self.r3d = []
        self.mdefs = []
        self._halo_tags = []
        self.subhalo_flags = []

        for halo in self.halos:
            self.masses.append(halo.mass)
            self.x.append(halo.x)
            self.y.append(halo.y)
            self.redshifts.append(halo.z)
            self.r2d.append(halo.r2d)
            self.r3d.append(halo.r3d)
            self.mdefs.append(halo.mdef)
            self._halo_tags.append(halo._unique_tag)
            self.subhalo_flags.append(halo._is_main_subhalo)

        self.masses = np.array(self.masses)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.r2d = np.array(self.r2d)
        self.r3d = np.array(self.r3d)
        self.redshifts = np.array(self.redshifts)

        self.unique_redshifts = np.unique(self.redshifts)

    def shift_background_to_source(self, source_x, source_y):

        """

        :param source_x: estimated angular source x coordinate [arcsec]
        :param source_y: estimated angular source y coordinate [arcsec]
        :param center_x_lens: lens centroid x [arcsec]
        :param center_y_lens: lens centroid y [arcsec]
        :param source_redshift: source redshift
        :return: an instance of realization in which all halos behind the
        main deflector are shifted such that the rendering volume closes at the
        offset source position
        """

        # add all halos in front of main deflector with positions unchanged
        halos = []
        background_halos = []

        for halo in self.halos:
            if halo.z <= self.geometry._zlens:
                halos.append(halo)
            else:
                background_halos.append(halo)

        background_lens_planes = self.unique_redshifts[
            np.where(self.unique_redshifts>self.geometry._zlens)]

        distance_calc = self.lens_cosmo.cosmo.D_C_transversez1z2

        Tz_lens = distance_calc(0, self.geometry._zlens)
        Tz_source = distance_calc(0, self.geometry._zsource)

        dT_perp_x_source = source_x * Tz_source
        dT_perp_y_source = source_y * Tz_source

        xlow, ylow = 0, 0

        shifted_background_halos = []

        for idx, zi in enumerate(background_lens_planes):

            Tz_current = distance_calc(0, zi)

            shiftx, shifty = self.geometry.interp_ray_angle(xlow, dT_perp_x_source,
                             ylow, dT_perp_y_source, Tz_lens, Tz_source, Tz_current)

            for halo in background_halos:
                if halo.z == zi:
                    new_x, new_y = halo.x + shiftx, halo.y + shifty
                    new_halo = Halo(mass=halo.mass, x=new_x, y=new_y, r2d=halo.r2d, r3d=halo.r3d, mdef=halo.mdef, z=halo.z,
                                    sub_flag=halo.is_subhalo, cosmo_m_prof=self.lens_cosmo, args=self._prof_params)
                    shifted_background_halos.append(new_halo)

        all_halos = halos + shifted_background_halos

        return Realization(None, None, None, None, None, None, None, None, self.halo_mass_function,
                           halos = all_halos, other_params= self._prof_params,
                           mass_sheet_correction=self._mass_sheet_correction)

    def _add_halo(self, m, x, y, r2, r3, md, z, sub_flag, halo=None):
        if halo is None:

            halo = Halo(mass=m, x=x, y=y, r2d=r2, r3d=r3, mdef=md, z=z, sub_flag=sub_flag, cosmo_m_prof=self.lens_cosmo,
                        args=self._prof_params)
        self._lensing_functions.append(self._lens(halo))
        self.halos.append(halo)

    def lensing_quantities(self, mass_sheet_correction_front=7.7,
                           mass_sheet_correction_back=8, return_kwargs=False):

        if self._overwrite_mass_sheet is not None:
            mass_sheet_correction_front = self._overwrite_mass_sheet
            mass_sheet_correction_back = self._overwrite_mass_sheet

        kwargs_lens = []
        lens_model_names = []
        redshift_list = []
        kwargs_lensmodel = None

        for i, halo in enumerate(self.halos):

            args = tuple([halo.x, halo.y, halo.mass, halo.z] + halo.profile_args)

            kw, model_args = self._lensing_functions[i].params(*args)

            lenstronomy_ID = self._lensing_functions[i].lenstronomy_ID

            lens_model_names.append(lenstronomy_ID)
            kwargs_lens.append(kw)
            redshift_list += [halo.z]

            if kwargs_lensmodel is None:
                kwargs_lensmodel = model_args
            else:

                if model_args is not None and not (type(model_args) is type(kwargs_lensmodel)):
                    raise Exception('Currently only one numerical lens class at once is supported.')

        if self._mass_sheet_correction:

            assert isinstance(mass_sheet_correction_front, float) or isinstance(mass_sheet_correction_front, int)
            assert mass_sheet_correction_front < 100, 'mass sheet correction should log(M)'
            assert isinstance(mass_sheet_correction_back, float) or isinstance(mass_sheet_correction_back, int)
            assert mass_sheet_correction_back < 100, 'mass sheet correction should log(M)'

            kwargs_mass_sheets, z_sheets = self.mass_sheet_correction(mlow_front=10**mass_sheet_correction_front,
                                                                      mlow_back=10**mass_sheet_correction_back)
            kwargs_lens += kwargs_mass_sheets
            lens_model_names += ['CONVERGENCE'] * len(kwargs_mass_sheets)
            redshift_list = np.append(redshift_list, z_sheets)

        if return_kwargs:
            return {'lens_model_list': lens_model_names,
                    'lens_redshift_list': redshift_list,
                    'z_source': self.geometry._zsource,
                    'z_lens': self.geometry._zlens,
                    'multi_plane': True}, kwargs_lens
        else:
            return lens_model_names, redshift_list, kwargs_lens, kwargs_lensmodel

    def _lens(self, halo):

        if halo.mdef not in self._loaded_models.keys():

            model = self._load_model(halo)
            self._loaded_models.update({halo.mdef: model})

        return self._loaded_models[halo.mdef]

    def _load_model(self, halo):

        if halo.mdef == 'NFW':
            from pyHalo.Lensing.NFW import NFWLensing
            lens = NFWLensing(self.lens_cosmo)

        elif halo.mdef == 'TNFW':
            from pyHalo.Lensing.NFW import TNFWLensing
            lens = TNFWLensing(self.lens_cosmo)

        elif halo.mdef == 'SIDM_TNFW':
            from pyHalo.Lensing.coredTNFW import coreTNFW
            lens = coreTNFW(self.lens_cosmo)

        elif halo.mdef == 'PT_MASS':
            from pyHalo.Lensing.PTmass import PTmassLensing
            lens = PTmassLensing(self.lens_cosmo)

        else:
            raise ValueError('halo profile ' + str(halo.mdef) + ' not recongnized.')

        return lens

    def _ray_position_z(self, thetax, thetay, zi, source_x, source_y):

        ray_angle_atz_x, ray_angle_atz_y = [], []

        for tx, ty in zip(thetax, thetay):

            if zi > self.geometry._zlens:
                angle_x_atz = self.geometry.ray_angle_atz(tx, zi, source_x)
                angle_y_atz = self.geometry.ray_angle_atz(ty, zi, source_y)
            else:
                angle_x_atz = self.geometry.ray_angle_atz(tx, zi)
                angle_y_atz = self.geometry.ray_angle_atz(ty, zi)

            ray_angle_atz_x.append(angle_x_atz)
            ray_angle_atz_y.append(angle_y_atz)

        return ray_angle_atz_x, ray_angle_atz_y

    def _interp_ray_angle_z(self, ray_comoving_x, ray_comoving_y, redshifts, zi, Tzlist, Tz_current):

        redshifts = np.array(redshifts)

        if zi in redshifts:
            angle_x, angle_y = [], []
            idx = np.where(redshifts == zi)[0][0].astype(int)
            for i in range(0, 4):

                angle_x.append(ray_comoving_x[idx][i] / Tzlist[idx])
                angle_y.append(ray_comoving_y[idx][i] / Tzlist[idx])
            angle_x = np.array(angle_x)
            angle_y = np.array(angle_y)

        else:

            adjacent_redshift_inds = np.argsort(np.absolute(redshifts - zi))[0:2]
            adjacent_redshifts = redshifts[adjacent_redshift_inds]

            if adjacent_redshifts[0] < adjacent_redshifts[1]:

                Tzlow, Tzhigh = Tzlist[adjacent_redshift_inds[0]], Tzlist[adjacent_redshift_inds[1]]
                xlow, xhigh = ray_comoving_x[adjacent_redshift_inds[0]], ray_comoving_x[adjacent_redshift_inds[1]]
                ylow, yhigh = ray_comoving_y[adjacent_redshift_inds[0]], ray_comoving_y[adjacent_redshift_inds[1]]
            else:

                Tzhigh, Tzlow = Tzlist[adjacent_redshift_inds[0]], Tzlist[adjacent_redshift_inds[1]]
                xhigh, xlow = ray_comoving_x[adjacent_redshift_inds[0]], ray_comoving_x[adjacent_redshift_inds[1]]
                yhigh, ylow = ray_comoving_y[adjacent_redshift_inds[0]], ray_comoving_y[adjacent_redshift_inds[1]]

            angle_x, angle_y = self.geometry.interp_ray_angle(xlow, xhigh, ylow, yhigh, Tzlow, Tzhigh, Tz_current)

        return angle_x, angle_y

    def filter_by_mass(self, mlow):

        halos = []
        for halo in self.halos:
            if halo.mass >= mlow:
                halos.append(halo)

        return Realization.from_halos(halos, self.halo_mass_function,
                                      self._prof_params, self._mass_sheet_correction)


    def filter(self, thetax, thetay, mindis_front=0.5, mindis_back=0.5, logmasscut_front=6, logmasscut_back=8,
               source_x=0, source_y=0, ray_x=None, ray_y=None,
               logabsolute_mass_cut_back=0, path_redshifts=None, path_Tzlist=None,
               logabsolute_mass_cut_front=0, centroid = [0, 0], zmin=None, zmax=None):

        halos = []

        thetax = thetax - centroid[0]
        thetay = thetay - centroid[1]

        if zmax is None:
            zmax = self.geometry._zsource
        if zmin is None:
            zmin = 0

        for plane_index, zi in enumerate(self.unique_redshifts):

            plane_halos = self.halos_at_z(zi)
            inds_at_z = np.where(self.redshifts == zi)[0]
            x_at_z = self.x[inds_at_z] - centroid[0]
            y_at_z = self.y[inds_at_z] - centroid[1]
            masses_at_z = self.masses[inds_at_z]

            if zi < zmin:
                continue
            if zi > zmax:
                continue

            if zi <= self.geometry._zlens:

                keep_inds_mass = np.where(masses_at_z >= 10 ** logmasscut_front)[0]

                inds_m_low = np.where(masses_at_z < 10 ** logmasscut_front)[0]

                keep_inds_dr = []

                for idx in inds_m_low:

                    for (anglex, angley) in zip(thetax, thetay):

                        dr = ((x_at_z[idx] - anglex) ** 2 +
                              (y_at_z[idx] - angley) ** 2) ** 0.5

                        if dr <= mindis_front:
                            keep_inds_dr.append(idx)
                            break

                keep_inds = np.append(keep_inds_mass, np.array(keep_inds_dr)).astype(int)

                if logabsolute_mass_cut_front > 0:
                    tempmasses = masses_at_z[keep_inds]
                    keep_inds = keep_inds[np.where(tempmasses >= 10 ** logabsolute_mass_cut_front)[0]]

            else:

                if ray_x is None or ray_y is None:
                    ray_at_zx, ray_at_zy = self._ray_position_z(thetax, thetay, zi, source_x, source_y)
                else:

                    Tz_current = self.geometry._cosmo.T_xy(0, zi)
                    ray_at_zx, ray_at_zy = self._interp_ray_angle_z(ray_x, ray_y,
                                                 path_redshifts, zi, path_Tzlist, Tz_current)

                keep_inds_mass = np.where(masses_at_z >= 10 ** logmasscut_back)[0]

                inds_m_low = np.where(masses_at_z < 10 ** logmasscut_back)[0]

                keep_inds_dr = []

                dr_list = []

                for idx in inds_m_low:

                    for (anglex, angley) in zip(ray_at_zx, ray_at_zy):

                        dr = ((x_at_z[idx] - anglex) ** 2 +
                              (y_at_z[idx] - angley) ** 2) ** 0.5

                        if dr <= mindis_back:
                            keep_inds_dr.append(idx)
                            dr_list.append(dr)
                            break

                keep_inds = np.append(keep_inds_mass, np.array(keep_inds_dr)).astype(int)

                if logabsolute_mass_cut_back > 0:
                    tempmasses = masses_at_z[keep_inds]
                    keep_inds = keep_inds[np.where(tempmasses >= 10 ** logabsolute_mass_cut_back)[0]]

            for halo_index in keep_inds:
                halos.append(plane_halos[halo_index])

        return Realization(None, None, None, None, None, None, None, None, self.halo_mass_function,
                           halos = halos, other_params= self._prof_params, mass_sheet_correction = self._mass_sheet_correction)

        #return Realization(None, None, None, None, None, None, None, None, self.halo_mass_function, halos=halos,
        #                   wdm_params=self._wdm_params, mass_sheet_correction=self._mass_sheet_correction)

    def mass_sheet_correction(self, mlow_front = 10**7.5, mlow_back = 10**8):

        kwargs = []
        zsheet = []
        unique_z = np.unique(self.redshifts)

        mhigh = 10**self._logmhigh

        mlow_front = 10**max(np.log10(mlow_front), self._logmlow)
        mlow_back = 10**max(np.log10(mlow_back), self._logmlow)

        kappa = None
        if len(unique_z) == 1 and unique_z[0] == self.geometry._zlens:
            if self._prof_params['subtract_subhalo_mass_sheet']:
                kappa = self.convergence_at_z(self.geometry._zlens,
                                              mlow_front, mhigh,None,self.m_break_scale,
                                              self.break_index,self.break_scale)
        if kappa is not None:
            kwargs.append({'kappa_ext': - self._prof_params['subhalo_mass_sheet_scale'] * kappa})
            zsheet.append(unique_z[0])

        for i in range(0, len(unique_z)-1):

            z = unique_z[i]

            delta_z = unique_z[i+1] - z

            kappa = None
            if z < self.geometry._zlens:
                kappa = self.convergence_at_z(z, mlow_front, mhigh, delta_z,
                                                     self.m_break_scale, self.break_index, self.break_scale)
            elif z > self.geometry._zlens:
                kappa = self.convergence_at_z(z, mlow_back, mhigh, delta_z,
                                                    self.m_break_scale, self.break_index, self.break_scale)

            if kappa is not None:
                kwargs.append({'kappa_ext': - self._prof_params['kappa_scale']*kappa})
                zsheet.append(z)

        return kwargs, zsheet

    def halos_at_z(self,z):
        halos = []
        for halo in self.halos:
            if halo.z != z:
                continue
            halos.append(halo)

        return halos

    def convergence_at_z(self, z, mlow, mhigh, delta_z, m_break, break_index, break_scale):

        if self._prof_params['subtract_exact_mass_sheets']:
            return self.convergence_at_z_exact(z)
        else:
            return self.convergence_at_z_theory(z, mlow, mhigh, delta_z, m_break, break_index, break_scale)

    def _sigma_crit_mass(self, z):

        area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, z)
        sigma_crit_mpc = self.lens_cosmo.get_epsiloncrit(z, self.geometry._zsource)

        return area * sigma_crit_mpc

    def convergence_at_z_theory(self, z, mlow, mhigh, delta_z, m_break, break_index, break_scale):

        if z == self.geometry._zlens:
            m_theory = self.mass_at_z_theory_lens(mlow, mhigh, m_break, break_index, break_scale)

        else:
            m_theory = self.mass_at_z_theory_LOS(z, delta_z, mlow, mhigh, m_break, break_index, break_scale)

        return m_theory / self._sigma_crit_mass(z)

    def convergence_at_z_exact(self, z):

        return self.mass_at_z_exact(z) / self._sigma_crit_mass(z)

    def mass_at_z_exact(self, z):

        inds = np.where(self.redshifts == z)
        m_exact = np.sum(self.masses[inds])
        return m_exact

    def mass_at_z_theory_lens(self, mlow, mhigh, m_break, break_index, break_scale):

        if 'sigma_sub' in self._prof_params.keys():
            sigma_sub = self._prof_params['sigma_sub']
            parent_m200 = self._prof_params['parent_m200']
            plaw_idx = self._prof_params['power_law_index']
            zlens = self.geometry._zlens
            kpc_per_asec_zlens = self.geometry._kpc_per_arcsec_zlens
            opening_angle = self._prof_params['cone_opening_angle']
            norm = norm_AO_from_sigmasub(sigma_sub, parent_m200,
                                         zlens, kpc_per_asec_zlens,
                                         opening_angle, plaw_idx)
        else:
            raise Exception('cannot subtract subhalo '
                            'mass sheets for this mass funciton.')

        def _integrand(m):
            return (norm * m ** plaw_idx) * \
                   (1 + break_scale*m_break/m) ** break_index

        m_theory = quad(_integrand, mlow, mhigh)[0]

        return m_theory

    def mass_at_z_theory_LOS(self, z, delta_z, mlow, mhigh, log_m_break, break_index, break_scale):

        mass = self.halo_mass_function.integrate_mass_function(z, delta_z, mlow, mhigh,
                         log_m_break, break_index, break_scale, n=1, norm_scale=self._LOS_norm)

        return mass

    def number_of_halos_before_redshift(self, z):
        n = 0
        for halo in self.halos:
            if halo.z <= z:
                n += 1
        return n

    def number_of_halos_after_redshift(self, z):
        n = 0
        for halo in self.halos:
            if halo.z > z:
                n += 1
        return n

    def number_of_halos_at_redshift(self, z):
        n = 0
        for halo in self.halos:
            if halo.z == z:
                n += 1
        return n

class RealizationFast(Realization):

    """
    A quick and dirty class useful for generating a realization with a few
    user-specified halos.
    """

    def __init__(self, masses, x, y, r2d, r3d, mdefs, z, subhalo_flag, zlens, zsource,
                 cone_opening_angle, log_mlow=6, log_mhigh=10, mass_sheet_correction=False):


        mfunc = LensingMassFunction(Cosmology(), 10**6, 10**10, zlens,
                                    zsource, cone_opening_angle)

        tup = (masses, x, y, r2d, r3d, mdefs, z, subhalo_flag)

        _aslist = False
        for element in tup:
            if isinstance(element, list) or isinstance(element, np.ndarray):
                _aslist = True
                break

        if _aslist:
            for element in tup:
                assert isinstance(element, list) or isinstance(element, np.ndarray), \
                    'All arguments must be either lists or floats.'

        else:
            tup = ([masses], [x], [y], [r2d], [r3d], [mdefs], [z], [subhalo_flag])

        default_params = {'cone_opening_angle': cone_opening_angle, 'opening_angle_factor': 6,
                          'log_mlow': log_mlow, 'log_mhigh': log_mhigh}

        Realization.__init__(self, *tup, halo_mass_function=mfunc,
                 halos = None, other_params = default_params,
                             mass_sheet_correction = mass_sheet_correction)

