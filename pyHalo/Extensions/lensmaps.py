from lenstronomy.LensModel.lens_model import LensModel
import numpy as np

class LensCone(object):

    def __init__(self, single_realization, z_source, mass_sheet_front=6, mass_sheet_back=6):

        self._realization = single_realization

        lens_model_names, redshift_list, kwargs_lens, kwargs_lensmodel = \
            single_realization.lensing_quantities(mass_sheet_front, mass_sheet_back)

        self._kwargs_lens = kwargs_lens
        self.lensmodel = LensModel(lens_model_list=lens_model_names, lens_redshift_list=redshift_list,
                                   multi_plane=True, z_source=z_source)

    def _Tz(self, z):

        return self.lensmodel.lens_model._cosmo_bkg.T_xy(0, z)

    def fermat_potential(self, x, y):

        pot = self.lensmodel.po

    def first_derivatives_point(self, x, y):

        defx, defy = self.lensmodel.alpha(x, y, self._kwargs_lens)

        return defx, defy

    def second_derivatives_point(self, x, y):

        fxx, fxy, fyx, fyy = self.lensmodel.hessian(x, y, self._kwargs_lens)

        return fxx, fxy, fyx, fyy

    def first_derivatives_grid(self, xmin, xmax, ymin, ymax, nsteps=50):

        x = np.linspace(xmin, xmax, nsteps)
        y = np.linspace(ymin, ymax, nsteps)
        xx, yy = np.meshgrid(x, y)
        shape0 = xx.shape

        defx, defy = self.lensmodel.alpha(xx.ravel(), yy.ravel(), self._kwargs_lens)

        return defx.reshape(shape0), defy.reshape(shape0)

    def second_derivatives_grid(self, xmin, xmax, ymin, ymax, nsteps=50):
        x = np.linspace(xmin, xmax, nsteps)
        y = np.linspace(ymin, ymax, nsteps)
        xx, yy = np.meshgrid(x, y)
        shape0 = xx.shape

        fxx, fxy, fyx, fyy = self.lensmodel.hessian(xx.ravel(), yy.ravel(), self._kwargs_lens)

        return fxx.reshape(shape0), fxy.reshape(shape0), fyx.reshape(shape0), fyy.reshape(shape0)

    def angular_position_at_z(self, x_obs, y_obs, z):

        x, y, _, _ = self.lensmodel.lens_model.ray_shooting_partial(0, 0, x_obs, y_obs, 0, z,
                                                                    self._kwargs_lens)

        Tz = self._Tz(z)

        return x/Tz, y/Tz


