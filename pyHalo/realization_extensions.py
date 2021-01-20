import numpy
import numpy as np
from pyHalo.Halos.HaloModels.PsuedoJaffe import PJaffeSubhalo
from pyHalo.single_realization import Realization


class RealizationExtensions(object):

    """
    This class supports operations that modify individual instances of the class Realization
    (see pyHalo.single_realization).
    """

    def __init__(self, realization):

        """

        :param realization: an instance of Realization
        """

        self._realization = realization

    def add_core_collapsed_subhalos(self, f_collapsed):
        """
        This function turns a fraction of subhalos in a realization into profiles modeled as PseudoJaffe profiles
        with 1/r^2 central density profiles. The new profile has the same total mass as the original NFW (or whatever)
        profile, but it's mass is redistributed accordingly. See documentation in PseudoJaffe profile class for more
        info.

        :param f_collapsed: fraction of subhalos that become isothermal profiles
        :return: A new instance of Realization where a fraction f_collapsed of the subhalos
        in the original realization have their mass definitions changed to PsuedoJaffe
        """

        halos = self._realization.halos

        for index, halo in enumerate(halos):
            if halo.is_subhalo:
                u = np.random.rand()
                if u < f_collapsed:
                    # change mass definition
                    new_halo = PJaffeSubhalo(halo.mass, halo.x, halo.y, halo.r3d, halo.mdef,
                                             halo.z, True, halo.lens_cosmo, halo._args, halo.unique_tag)
                    halos[index] = new_halo

        lens_cosmo = self._realization.lens_cosmo
        prof_params = self._realization._prof_params
        msheet_correction = self._realization._mass_sheet_correction
        rendering_classes = self._realization.rendering_classes
        rendering_center_x, rendering_center_y = self._realization.rendering_center

        return Realization.from_halos(halos, lens_cosmo, prof_params,
                                      msheet_correction, rendering_classes,
                                      rendering_center_x, rendering_center_y,
                                      self._realization.geometry)
