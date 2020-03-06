from copy import deepcopy

import numpy as np

from pyHalo.Rendering.Field.base import LOSBase
from pyHalo.Rendering.parameterizations import BrokenPowerLaw
from pyHalo.Rendering.keywords import LOS_powerlaw_mfunc

from pyHalo.Spatial.uniform import Uniform

class LOSPowerLawDynamic(LOSBase):

    def __init__(self, args, lensing_mass_func, lens_plane_redshifts, delta_zs,
                 aperture_x, aperture_y, aperture_size):

        self._rendering_args = LOS_powerlaw_mfunc(args, lensing_mass_func)

        self._lens_plane_redshifts, self._delta_zs = lens_plane_redshifts, delta_zs
        self._zlens = lensing_mass_func.geometry._zlens
        spatial_parameterization = Uniform(aperture_size, lensing_mass_func.geometry)

        self._aperture_size, self._aperture_x, self._aperture_y = aperture_size, aperture_x, aperture_y

        super(LOSPowerLawDynamic, self).__init__(lensing_mass_func, self._rendering_args,
                                          spatial_parameterization)

    def _volume_element_comoving(self, radius_arcsec, z, delta_z):

        cosmo = self._geometry._cosmo

        dr_comoving = self._geometry.delta_R_comoving(z, delta_z)
        radius_radian = radius_arcsec * cosmo.arcsec
        radius_comoving = radius_radian * cosmo.D_C(z)

        return np.pi * radius_comoving ** 2 * dr_comoving

    def _angle_scale(self, z, zref):

        if z <= zref:
            return 1.
        else:
            cosmo = self._geometry._cosmo
            return cosmo.D_C(zref)/cosmo.D_C(z)

    def __call__(self):

        add_two_halo_term = self._lensing_mass_func._two_halo_term
        redshifts = []

        assert len(self._lens_plane_redshifts) == len(self._delta_zs)
        assert len(self._aperture_x) == len(self._lens_plane_redshifts)

        for i, (zi, delta_zi, x_c, y_c) in enumerate(zip(self._lens_plane_redshifts, self._delta_zs,
                                                         self._aperture_x, self._aperture_y)):

            if zi < self._rendering_args['zmin'] or zi > self._rendering_args['zmax']:

                m, x, y, r2, r3 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
                redshifts += []

            else:
                boost = 1.

                if zi == self._zlens:
                    if add_two_halo_term:
                        rmax = 20 #Mpc
                        boost = self._lensing_mass_func.two_halo_boost(self._rendering_args['parent_m200'], zi, rmax=rmax)

                norm_dV = self._rendering_args['LOS_normalization'] * boost * self._lensing_mass_func.norm_at_z_density(zi)

                rescale_angle = self._angle_scale(zi, self._zlens)

                aperture_size = self._aperture_size * rescale_angle
                volume_element = self._volume_element_comoving(aperture_size, zi, delta_zi)

                norm = norm_dV * volume_element
                pargs = deepcopy(self._rendering_args)
                plaw_index = self._lensing_mass_func.plaw_index_z(zi)

                pargs.update({'normalization': norm})
                pargs.update({'power_law_index': plaw_index})
                mfunc = BrokenPowerLaw(**pargs)

                m = mfunc.draw()
                x, y, r2, r3 = self.render_positions_at_z(zi, len(m), rescale_angle)

                x += x_c
                y += y_c
                redshifts += [zi] * len(x)

            if i == 0:
                masses = m
                x_arcsec, y_arcsec, r2d, r3d = x, y, r2, r3
            else:
                x_arcsec = np.append(x_arcsec, x)
                y_arcsec = np.append(y_arcsec, y)
                masses = np.append(masses, m)
                r2d = np.append(r2d, r2)
                r3d = np.append(r3d, r3)

        return masses, x_arcsec, y_arcsec, r2d, r3d, np.array(redshifts)
