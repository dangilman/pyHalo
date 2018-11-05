import numpy as np
from pyHalo.Lensing.NFW import NFWLensing
from pyHalo.Lensing.TNFW import TNFWLensing
from pyHalo.Lensing.PTmass import PTmassLensing
from pyHalo.Lensing.PJaffe import PJaffeLensing
from pyHalo.defaults import default_z_step

def realization_at_z(realization,z):

    halos = realization.halos_at_z(z)

    halo_mass_function, wdm_params = realization.halo_mass_function, realization._wdm_params

    return Realization(None, None, None, None, None, None, None, None, halo_mass_function, halos=halos,
                           wdm_params=wdm_params)

class Halo(object):

    def __init__(self, mass, x, y, r2d, r3d, mdef, z, args):

        self.mass = mass
        self.x = x
        self.y = y
        self.r2d = r2d
        self.r3d = r3d
        self.mdef = mdef
        self.z = z
        self.mass_def_arg = args
        self._unique_tag = np.random.rand()

class Realization(object):

    #max_m_high = 10**9

    def __init__(self, masses, x, y, r2d, r3d, mdefs, z, mass_def_args, halo_mass_function,
                 halos = None, wdm_params = None, mass_sheet_correction = True):

        self._mass_sheet_correction  = mass_sheet_correction
        self._subtract_theory_mass_sheets = True
        self._overwrite_mass_sheet = None
        self._kappa_scale = 1
        #self._kappa_scale = 1.269695
        # 1.269695 for TNFW halos truncated at r50

        self.halo_mass_function = halo_mass_function
        self.geometry = halo_mass_function.geometry
        self.lens_cosmo = self.geometry._lens_cosmo
        self._lensing_functions = []
        self.halos = []

        if wdm_params is None:
            self.m_break_scale = 0
            self.break_index = -1.3
            self._LOS_norm = 1
        else:
            self.m_break_scale = wdm_params['log_m_break']
            self.break_index = wdm_params['break_index']
            if 'LOS_normalization' in wdm_params:
                self._LOS_norm = wdm_params['LOS_normalization']
            else:
                self._LOS_norm = 1

        self._wdm_params = wdm_params

        if halos is None:

            for mi, xi, yi, r2di, r3di, mdefi, zi, mdefargi in zip(masses, x, y, r2d, r3d,
                           mdefs, z, mass_def_args):

                self._add_halo(mi, xi, yi, r2di, r3di, mdefi, zi, mdefargi)
        else:

            for halo in halos:
                self._add_halo(None, None, None, None, None, None, None, None, halo)

        self._reset()

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

        return Realization(None, None, None, None, None, None, None, None, self.halo_mass_function, halos=halos,
                           wdm_params=self._wdm_params, mass_sheet_correction=self._mass_sheet_correction)

    def _reset(self):

        self.x = []
        self.y = []
        self.masses = []
        self.redshifts = []
        self.r2d = []
        self.r3d = []
        self.mdefs = []
        self.mass_def_args = []
        self._halo_tags = []

        for halo in self.halos:
            self.masses.append(halo.mass)
            self.x.append(halo.x)
            self.y.append(halo.y)
            self.redshifts.append(halo.z)
            self.r2d.append(halo.r2d)
            self.r3d.append(halo.r3d)
            self.mdefs.append(halo.mdef)
            self.mass_def_args.append(halo.mass_def_arg)
            self._halo_tags.append(halo._unique_tag)

        self.masses = np.array(self.masses)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.r2d = np.array(self.r2d)
        self.r3d = np.array(self.r3d)
        self.redshifts = np.array(self.redshifts)

        self._unique_redshifts = np.unique(self.redshifts)

    def _add_halo(self, m, x, y, r2, r3, md, z, mdarg, halo=None):
        if halo is None:
            halo = Halo(m, x, y, r2, r3, md, z, mdarg)
        self._lensing_functions.append(self._lens(halo))
        self.halos.append(halo)

    def lensing_quantities(self, mass_sheet_correction = 8):

        if self._overwrite_mass_sheet is not None:
            mass_sheet_correction = self._overwrite_mass_sheet

        kwargs_lens = []
        lens_model_names = []

        for i, halo in enumerate(self.halos):

            args = {'x': halo.x, 'y': halo.y, 'mass': halo.mass}
            lens_model_names.append(halo.mdef)

            if halo.mdef == 'NFW':
                args.update({'concentration':halo.mass_def_arg['concentration'],'redshift':halo.z})
            elif halo.mdef == 'TNFW':
                args.update({'concentration': halo.mass_def_arg['concentration'], 'redshift': halo.z})
                args.update({'r_trunc': halo.mass_def_arg['r_trunc']})
            elif halo.mdef == 'POINT_MASS':
                args.update({'redshift': halo.z})
            elif halo.mdef == 'PJAFFE':
                args.update({'r_trunc': halo.mass_def_arg['r_trunc']})
            else:
                raise ValueError('halo profile '+str(halo.mdef)+' not recongnized.')

            kwargs_lens.append(self._lensing_functions[i].params(**args))

        if self._mass_sheet_correction and mass_sheet_correction is not False:

            assert isinstance(mass_sheet_correction, float) or isinstance(mass_sheet_correction, int)
            assert mass_sheet_correction < 10, 'mass sheet correction should log(M)'

            kwargs_mass_sheets, z_sheets = self.mass_sheet_correction(mlow = 10**mass_sheet_correction)
            kwargs_lens += kwargs_mass_sheets
            lens_model_names += ['CONVERGENCE'] * len(kwargs_mass_sheets)
            redshift_list = np.append(self.redshifts, z_sheets)

        else:
            redshift_list = self.redshifts

        return lens_model_names, redshift_list, kwargs_lens

    def _lens(self, halo):

        if halo.mdef == 'NFW':
            lens = NFWLensing(self.lens_cosmo)

        elif halo.mdef == 'TNFW':
            lens = TNFWLensing(self.lens_cosmo)

        elif halo.mdef == 'POINT_MASS':
            lens = PTmassLensing(self.lens_cosmo)

        elif halo.mdef == 'PJAFFE':
            lens = PJaffeLensing(self.lens_cosmo)

        else:
            raise ValueError('halo profile ' + str(halo.mdef) + ' not recongnized.')

        return lens

    def _ray_position_z(self, thetax, thetay, zi, source_x, source_y):

        ray_angle_atz_x, ray_angle_atz_y = [], []

        for tx, ty in zip(thetax, thetay):

            angle_x_atz = self.geometry.ray_angle_atz(tx, zi, self.geometry._zlens)
            angle_y_atz = self.geometry.ray_angle_atz(ty, zi, self.geometry._zlens)

            if zi > self.geometry._zlens:
                angle_x_atz += source_x
                angle_y_atz += source_y

            ray_angle_atz_x.append(angle_x_atz)
            ray_angle_atz_y.append(angle_y_atz)

        return ray_angle_atz_x, ray_angle_atz_y

    def _interp_ray_angle_z(self, background_redshifts, Tzlist_background,
                            ray_x, ray_y, zi, thetax, thetay):

        angle_x, angle_y = [], []

        if zi in background_redshifts:

            idx = np.where(background_redshifts == zi)[0][0].astype(int)

            for i, (tx, ty) in enumerate(zip(thetax, thetay)):

                angle_x.append(ray_x[idx][i] / Tzlist_background[idx])
                angle_y.append(ray_y[idx][i] / Tzlist_background[idx])

        else:

            ind_low = np.where(background_redshifts - zi < 0)[0][-1].astype(int)
            ind_high = np.where(background_redshifts - zi > 0)[0][0].astype(int)

            Tz = self.geometry._cosmo.T_xy(0, zi)

            for i in range(0, len(thetax)):

                x0 = Tzlist_background[ind_low]
                bx = ray_x[ind_low][i]
                by = ray_y[ind_low][i]

                run = (Tzlist_background[ind_high] -x0)
                slopex = (ray_x[ind_high][i] - bx) * run ** -1
                slopey = (ray_y[ind_high][i] - by) * run ** -1

                delta_x = Tz - x0

                newx = slopex * delta_x + bx
                newy = slopey * delta_x + by

                angle_x.append(newx / Tz)
                angle_y.append(newy / Tz)

        return np.array(angle_x), np.array(angle_y)

    def filter(self, thetax, thetay, mindis_front=0.5, mindis_back=0.5, logmasscut_front=6, logmasscut_back=8,
               source_x=0, source_y=0, ray_x=None, ray_y=None,
               logabsolute_mass_cut_back=0, path_redshifts=None, path_Tzlist=None,
               logabsolute_mass_cut_front=0):

        halos = []

        for plane_index, zi in enumerate(self._unique_redshifts):

            plane_halos = self.halos_at_z(zi)
            inds_at_z = np.where(self.redshifts == zi)[0]
            x_at_z = self.x[inds_at_z]
            y_at_z = self.y[inds_at_z]
            masses_at_z = self.masses[inds_at_z]

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
                    ray_at_zx, ray_at_zy = self._interp_ray_angle_z(path_redshifts, path_Tzlist, ray_x,
                                                                    ray_y,
                                                                    zi, thetax, thetay)

                keep_inds_mass = np.where(masses_at_z >= 10 ** logmasscut_back)[0]

                inds_m_low = np.where(masses_at_z < 10 ** logmasscut_back)[0]

                keep_inds_dr = []

                for idx in inds_m_low:

                    for (anglex, angley) in zip(ray_at_zx, ray_at_zy):

                        dr = ((x_at_z[idx] - anglex) ** 2 +
                              (y_at_z[idx] - angley) ** 2) ** 0.5

                        if dr <= mindis_back:
                            keep_inds_dr.append(idx)
                            break

                keep_inds = np.append(keep_inds_mass, np.array(keep_inds_dr)).astype(int)

                if logabsolute_mass_cut_back > 0:
                    tempmasses = masses_at_z[keep_inds]
                    keep_inds = keep_inds[np.where(tempmasses >= 10 ** logabsolute_mass_cut_back)[0]]

            for halo_index in keep_inds:
                halos.append(plane_halos[halo_index])

        return Realization(None, None, None, None, None, None, None, None, self.halo_mass_function, halos=halos,
                           wdm_params=self._wdm_params, mass_sheet_correction=self._mass_sheet_correction)

    def mass_sheet_correction(self, mlow = 10**8, mhigh = 10**10):

        kwargs = []
        zsheet = []
        unique_z = np.unique(self.redshifts)

        for i in range(0, len(unique_z) - 1):

            z = unique_z[i]
            delta_z = unique_z[i+1] - z

            if z != self.geometry._zlens:

                if self._subtract_theory_mass_sheets:

                    kappa = self.convergence_at_z_theory(z, mlow, mhigh, delta_z, self.m_break_scale, self.break_index)
                else:
                    kappa = self.convergence_at_z(z, 0)

                if kappa > 0:
                    #kwargs.append({'kappa_ext': - 1.269695*kappa})
                    kwargs.append({'kappa_ext': - self._kappa_scale * kappa})
                    zsheet.append(z)

        return kwargs, zsheet

    def halos_at_z(self,z):
        halos = []
        for halo in self.halos:
            if halo.z != z:
                continue
            halos.append(halo)

        return halos

    def _convergence_at_z(self, m_rendered, z):

        area = self.geometry._angle_to_arcsec_area(self.geometry._zlens, z)
        scrit = self.geometry._lens_cosmo.get_sigmacrit(z)

        return m_rendered / area / scrit

    def convergence_at_z_theory(self, z, mlow, mhigh, delta_z, m_break, break_index):

        m_theory = self.mass_at_z_theory(z, delta_z, mlow, mhigh, m_break, break_index)

        return self._convergence_at_z(m_theory, z)

    def convergence_at_z(self, z, m_scale):

        m = self.mass_at_z(z, m_scale)

        return self._convergence_at_z(m, z)

    def mass_at_z_theory(self, z, delta_z, mlow, mhigh, log_m_break, break_index):

        #m_rendered = self.mass_at_z(z, log_m_break)
        #if m_rendered > 0:
        #    mhigh = min(m_rendered, mhigh)

        mass = self.halo_mass_function.integrate_mass_function(z, delta_z, mlow, mhigh, log_m_break,
                                                               break_index, norm_scale=self._LOS_norm)

        return mass

    def mass_at_z(self,z, logmcut = 0):

        def z_is_close(z1, z2):
            return np.absolute(z2 - z1) < default_z_step * 0.01

        mass = 0

        for i, mi in enumerate(self.masses):

            if z_is_close(self.redshifts[i], z):

                if np.log10(mi) >= logmcut:
                    mass += mi

        return mass
