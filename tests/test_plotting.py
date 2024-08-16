import numpy.testing as npt
import pytest
from pyHalo.PresetModels.cdm import CDM
from pyHalo.plotting_routines import *

class TestPlottingRoutines(object):

    def setup_method(self):
        self.realization = CDM(0.5, 1.5, sigma_sub=0.025, LOS_normalization=0.0)

    def test_spatial_distribution(self):
        plot_subhalo_spatial_distribution(self.realization)

    def test_mass_function_plot(self):

        plot_subhalo_mass_functon(self.realization)
        plot_subhalo_mass_functon(self.realization, bound_mass_function=True)
        plot_halo_mass_function(self.realization, z_eval=0.5)
        plot_halo_mass_function(self.realization, z_eval=None)
        plot_halo_mass_function(self.realization, z_eval=[0.5, 0.6])

    def test_bound_mass_function_plot(self):

        plot_subhalo_bound_mass(self.realization)
        plot_bound_mass_histogram(self.realization)
        plot_subhalo_concentration_versus_bound_mass(self.realization)
        plot_subhalo_infall_time_versus_bound_mass(self.realization)

    def test_plot_mc_relation(self):

        plot_concentration_mass_relation(self.realization, z_eval='z_lens')
        plot_concentration_mass_relation(self.realization, z_eval=0.5)
        plot_concentration_mass_relation(self.realization, z_eval=[0.4, 0.6])

    def test_truncation_radius_plot(self):

        plot_truncation_radius_histogram(self.realization, subhalos_only=False)
        plot_truncation_radius_histogram(self.realization, subhalos_only=True)

    def test_convergence_plot(self):

        fig = plt.figure(1)
        ax = plt.subplot(111)
        plot_multiplane_convergence(self.realization, ax=ax, npix=10, show_critical_curve=False)
        plot_multiplane_convergence(self.realization, npix=10, show_critical_curve=False, subtract_mean_kappa=True)
        plot_multiplane_convergence(self.realization, npix=10, show_critical_curve=True)

if __name__ == '__main__':
     pytest.main()
