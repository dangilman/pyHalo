import numpy as np
from pyHalo.Rendering.KappaSheets.base import KappaSheetBase


class MeanFieldDeltaFunction(KappaSheetBase):

    def __call__(self, realization=None):

        if 'subtract_exact_mass_sheets' in self.kwargs_rendering:

            return self.convergence_in_planes_exact(realization)

        else:

            return self.convergence_in_planes_theory()

    def convergence_in_planes_theory(self):

        mass_fraction = self.kwargs_rendering['mass_fraction']

        mass_in_planes = self.mass_in_planes_theory(mass_fraction)

        mass_in_planes *= self.kwargs_rendering['kappa_scale']

        return self.negative_kappa_sheets(mass_in_planes)

    def mass_in_planes_theory(self, mass_fraction):

        masses = []

        for z, delta_z in zip(self.redshifts, self.delta_zs):

            rho_dV = self.halo_mass_function.rho_dV(mass_fraction)
            M_theory = rho_dV * self.geometry.volume_element_comoving(z, delta_z)

            masses.append(M_theory)

        return np.array(masses)
