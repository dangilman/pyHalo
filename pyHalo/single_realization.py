import numpy as np
from pyHalo.Lensing.NFW import NFWLensing
from pyHalo.Lensing.TNFW import TNFWLensing
from pyHalo.Lensing.PTmass import PTmassLensing
from pyHalo.Lensing.PJaffe import PJaffeLensing

def realization_at_z(realization,z):

    m, x, y, r2, r3, mdef, redshift, mdefargs = realization.halos_at_z(z)

    return Realization(m, x, y, r2, r3, mdef, redshift, mdefargs, realization.geometry)

class Realization(object):

    def __init__(self, masses, x, y, r2d, r3d, mdefs, z, mass_def_args, geometry):

        self.masses = masses
        self.x = x
        self.y = y
        self.mdefs = mdefs
        self.redshifts = z
        self.mass_def_args = mass_def_args

        self.r2d = r2d
        self.r3d = r3d

        self.geometry = geometry
        self.lens_cosmo = geometry._lens_cosmo
        self._unique_redshifts = np.unique(self.redshifts)

        self._lensing_functions = self._lensing_list()

    def lensing_quantities(self, mass_sheet_correction = True):

        kwargs_lens = []
        lens_model_names = []

        for i,mdef in enumerate(self.mdefs):

            args = {'x': self.x[i], 'y': self.y[i], 'mass': self.masses[i]}
            lens_model_names.append(mdef)

            if mdef == 'NFW':
                args.update({'concentration':self.mass_def_args[i]['concentration'],'redshift':self.redshifts[i]})
            elif mdef == 'TNFW':
                args.update({'concentration': self.mass_def_args[i]['concentration'], 'redshift': self.redshifts[i]})
                args.update({'r_trunc': self.mass_def_args[i]['r_trunc']})
            elif mdef == 'POINT_MASS':
                args.update({'redshift': self.redshifts[i]})
            elif mdef == 'PJAFFE':
                args.update({'r_trunc': self.mass_def_args[i]['r_trunc']})
            else:
                raise ValueError('halo profile '+str(mdef)+' not recongnized.')

            kwargs_lens.append(self._lensing_functions[i].params(**args))

        if mass_sheet_correction:

            kwargs_mass_sheets, z_sheets = self.mass_sheet_correction()
            kwargs_lens += kwargs_mass_sheets
            lens_model_names += ['CONVERGENCE'] * len(kwargs_mass_sheets)
            redshift_list = np.append(self.redshifts, z_sheets)

        else:
            redshift_list = self.redshifts

        return lens_model_names, redshift_list, kwargs_lens

    def _lensing_list(self):

        lensing = []

        for i, mdef in enumerate(self.mdefs):

            if mdef == 'NFW':
                lensing.append(NFWLensing(self.lens_cosmo))

            elif mdef == 'TNFW':
                lensing.append(TNFWLensing(self.lens_cosmo))

            elif mdef == 'POINT_MASS':
                lensing.append(PTmassLensing(self.lens_cosmo))

            elif mdef == 'PJAFFE':
                lensing.append(PJaffeLensing(self.lens_cosmo))

            else:
                raise ValueError('halo profile ' + str(mdef) + ' not recongnized.')

        return lensing

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

    def filter(self, thetax, thetay, mindis_front = 0.5, mindis_back = 0.5, logmasscut_front = 6, logmasscut_back = 8,
               source_x = 0, source_y = 0, ray_x = None, ray_y = None,
               logabsolute_mass_cut = 0, background_redshifts = None, Tzlist_background = None):

        masses, x, y, mdefs, mdef_args, r2d, r3d, redshifts = [], [], [], [], [], [], [], []
        start = True

        for plane_index, zi in enumerate(self._unique_redshifts):

            inds_at_z = np.where(self.redshifts == zi)[0]
            x_at_z = self.x[inds_at_z]
            y_at_z = self.y[inds_at_z]
            masses_at_z = self.masses[inds_at_z]

            mdefs_z = [self.mdefs[idx] for idx in inds_at_z]
            mdef_args_z = [self.mass_def_args[idx] for idx in inds_at_z]

            r2dz = self.r2d[inds_at_z]
            r3dz = self.r3d[inds_at_z]

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
            else:

                if ray_x is None or ray_y is None:
                    ray_at_zx, ray_at_zy = self._ray_position_z(thetax, thetay, zi, source_x, source_y)
                else:
                    ray_at_zx, ray_at_zy = self._interp_ray_angle_z(background_redshifts, Tzlist_background, ray_x, ray_y,
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

            if logabsolute_mass_cut > 0:
                tempmasses = masses_at_z[keep_inds]
                keep_inds = keep_inds[np.where(tempmasses >= 10**logabsolute_mass_cut)[0]]


            if start:

                masses = np.array(masses_at_z[keep_inds])
                x = np.array(x_at_z[keep_inds])
                y = np.array(y_at_z[keep_inds])
                r2d = np.array(r2dz[keep_inds])
                r3d = np.array(r3dz[keep_inds])
                redshifts = np.array([zi]*len(keep_inds))
                start = False

            else:

                masses = np.append(masses, np.array(masses_at_z[keep_inds]))
                x = np.append(x, np.array(x_at_z[keep_inds]))
                y = np.append(y, np.array(y_at_z[keep_inds]))
                r2d = np.append(r2d, np.array(r2dz[keep_inds]))
                r3d = np.append(r3d, np.array(r3dz[keep_inds]))
                redshifts = np.append(redshifts, np.array([zi] * len(keep_inds)))

            mdefs += [mdefs_z[idx] for idx in keep_inds]
            mdef_args += [mdef_args_z[idx] for idx in keep_inds]

        return Realization(np.array(masses), np.array(x), np.array(y), np.array(r2d), np.array(r3d), mdefs, redshifts,
                                        mdef_args, self.geometry)

    def mass_sheet_correction(self):

        kwargs = []
        zsheet = []

        for z in self._unique_redshifts:

            if z != self.geometry._zlens:

                kappa = self.convergence_at_z(z)
                kwargs.append({'kappa_ext': - kappa})
                zsheet.append(z)

        return kwargs, zsheet

    def halos_at_z(self,z):

        masses, x, y, r2d, r3d, mdefs, massdefargs = [], [], [], [], [], [], []

        for i, mdef in enumerate(self.mdefs):
            if self.redshifts[i] != z:
                continue

            masses.append(self.masses[i])
            x.append(self.x[i])
            y.append(self.y[i])
            r2d.append(self.r2d[i])
            r3d.append(self.r3d[i])
            mdefs.append(self.mdefs[i])

            massdefargs.append(self.mass_def_args[i])

        return np.array(masses), np.array(x), np.array(y), np.array(r2d), np.array(r3d), mdefs, np.array([z]*len(masses)), \
               massdefargs

    def convergence_at_z(self,z):

        m = self.mass_at_z(z)

        area = self.geometry._angle_to_arcsec_area(self.geometry._zlens, z)

        sigmacrit = self.geometry._lens_cosmo.get_sigmacrit(z)

        kappa = m / area / sigmacrit

        return kappa

    def mass_at_z(self,z):

        mass = 0

        for i, mi in enumerate(self.masses):
            if self.redshifts[i] == z:
                mass += mi

        return mass
