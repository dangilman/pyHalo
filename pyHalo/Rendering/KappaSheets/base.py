from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad
import numpy as np
from pyHalo.Halos.lens_cosmo import LensCosmo

class KappaSheetBase(object):

    def __init__(self, pyhalo_instance, kwargs_rendering, log_mlow, log_mhigh,
                 normalization_function=None):

        self.halo_mass_function = pyhalo_instance.build_LOS_mass_function(kwargs_rendering)

        self.geometry = self.halo_mass_function.geometry

        self.kwargs_rendering = kwargs_rendering

        self.redshifts, self.delta_zs = pyhalo_instance.lens_plane_redshifts(kwargs_rendering)

        self._log_mlow, self._log_mhigh = log_mlow, log_mhigh

        self.lens_cosmo = LensCosmo(self.geometry._zlens, self.geometry._zsource,
                                    self.geometry._cosmo)

        self.normalization_function = normalization_function

    def convergence_in_planes_exact(self, realization):

        mass_in_planes = self.mass_in_planes_exact(realization)

        return self.negative_kappa_sheets(mass_in_planes)

    def mass_in_planes_exact(self, realization):

        masses = []

        for z in self.redshifts:
            mass = realization.mass_at_z_exact(z)
            masses.append(mass)

        return np.array(masses)

    def negative_kappa_sheets(self, mass_in_planes):

        negative_kappa = []

        for mass, z in zip(mass_in_planes, self.redshifts):
            sigma_crit_mass = self.sigma_crit_mass(z)
            kappa = mass / sigma_crit_mass
            negative_kappa.append(-kappa)

        return negative_kappa

    def sigma_crit_mass(self, z):
        area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, z)
        sigma_crit_mpc = self.lens_cosmo.get_epsiloncrit(z, self.geometry._zsource)

        return area * sigma_crit_mpc

