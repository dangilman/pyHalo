from pyHalo.pyhalo_base import pyHaloBase
from pyHalo.single_realization import Realization
from pyHalo.Rendering.halo_population import HaloPopulation
from pyHalo.defaults import set_default_kwargs
from pyHalo.Halos.lens_cosmo import LensCosmo


class pyHalo(pyHaloBase):

    def __init__(self, zlens, zsource, cosmology_kwargs={},
                 kwargs_halo_mass_function={}):

        """
        This class manages the creation of dark matter substructure realizations, coordinating the
        rendering of line-of-sight and subhalos in the lensing volume. For usage examples see
        the example notebooks in pyhalo/example_notebooks

        :param zlens: lens redshift
        :param zsource: source redshift
        :param cosmology_kwargs:
        keyword arguments for 'Cosmology' class. See documentation in cosmology.py
        :param halo_mass_function_args:
        keyword arguments for 'LensingMassFunction' class. See documentation in lensing_mass_function.py
        :param kwargs_massfunc:
        keyword arguments
        """
        super(pyHalo, self).__init__(zlens, zsource, cosmology_kwargs, kwargs_halo_mass_function)

    def render(self, population_model_list, model_keywords, nrealizations=1,
               convergence_sheet_correction=True):

        halo_mass_function = self.build_LOS_mass_function(model_keywords)
        geometry = self.halo_mass_function.geometry
        keywords_master = set_default_kwargs(model_keywords, self.zsource)

        lens_cosmo = LensCosmo(self.zlens, self.zsource, self.cosmology)
        plane_redshifts, redshift_spacing = self.lens_plane_redshifts(keywords_master)

        realization_list = []

        for n in range(nrealizations):

            population_model = HaloPopulation(population_model_list, keywords_master, lens_cosmo, geometry,
                                              halo_mass_function, plane_redshifts, redshift_spacing)

            masses, x_arcsec, y_arcsec, r3d, redshifts, subhalo_flag = population_model.render()

            mdefs = []
            for i in range(0, len(masses)):
                if subhalo_flag[i]:
                    mdefs += [keywords_master['mdef_subs']]
                else:
                    mdefs += [keywords_master['mdef_los']]

            realization = Realization(masses, x_arcsec, y_arcsec, r3d, mdefs, redshifts, subhalo_flag, self.halo_mass_function,
                                      halo_profile_args=keywords_master, mass_sheet_correction=convergence_sheet_correction,
                                      rendering_classes=population_model.rendering_classes)
            realization_list.append(realization)

        return realization_list
