import numpy.testing as npt
import pytest
from pyHalo.preset_models import CDM
from pyHalo.plotting_routines import plot_subhalo_bound_mass, \
    plot_subhalo_mass_functon, plot_concentration_mass_relation, plot_halo_mass_function

class TestPlottingRoutines(object):

    def setup_method(self):
        self.realization = CDM(0.5, 1.5, sigma_sub=0.025, LOS_normalization=0.0)

    def test_mass_function_plot(self):

        plot_subhalo_mass_functon(self.realization)
        plot_subhalo_mass_functon(self.realization, bound_mass_function=True)
        plot_halo_mass_function(self.realization, z_eval=0.5)
        plot_halo_mass_function(self.realization, z_eval=None)
        plot_halo_mass_function(self.realization, z_eval=[0.5, 0.6])

    def test_bound_mass_function_plot(self):

        plot_subhalo_bound_mass(self.realization)

    def test_plot_mc_relation(self):

        plot_concentration_mass_relation(self.realization, z_eval='z_lens')
        plot_concentration_mass_relation(self.realization, z_eval=0.5)
        plot_concentration_mass_relation(self.realization, z_eval=[0.4, 0.6])


if __name__ == '__main__':
     pytest.main()
