from pyHalo.Halos.lens_cosmo import LensCosmo

class RenderingBase(object):

    def __init__(self, geometry_class, lens_cosmo=None):

        self.geometry = geometry_class

        self._zlens = self.geometry._zlens

        self._zsource = self.geometry._zsource

        if lens_cosmo is None:
            lens_cosmo = LensCosmo(geometry_class._zlens, geometry_class._zsource, geometry_class._cosmo)

        self.lens_cosmo = lens_cosmo

    def render_positions_at_z(self, *args, **kwargs):

        raise Exception('This class does not yet have the routine "render_positions_at_z"')

    def render_masses(self, *args, **kwargs):

        raise Exception('This class does not yet have the routine "render_masses"')

    def negative_kappa_sheets_theory(self, *args, **kwargs):

        raise Exception('This class does not yet have the routine "negative_kappa_sheets_theory"')

    def keys_convergence_sheets(self, *args, **kwargs):

        raise Exception('This class does not yet have the routine "keys_convergence_sheets"')

    def keyword_parse(self, *args, **kwargs):

        raise Exception('This class does not yet have the routine "keyword_parse"')
