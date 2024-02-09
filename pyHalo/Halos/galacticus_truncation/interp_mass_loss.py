import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyHalo.Halos.galacticus_truncation.tabulated_mass_loss import _log10_mbound_over_minfall, \
    _log10_mbound_over_minfall_scatter_dex
from pyHalo.Halos.galacticus_truncation.johnsonSUpdf import a_JohnsonSU, b_JohnsonSU
from scipy.stats import johnsonsu

class InterpGalacticus(object):
    """
    This class interpolates output from the semi-analytic model galacticus to predict the bound mass of a subhalo
    as a function of its infall mass, concentration, host halo concentration, and the time since infall.
    """
    def __init__(self):
        _t_ellapsed_min, _t_ellapsed_max = 0.0, 7.658449372663094
        _chostmin, _chostmax = 2.0, 8.0
        _log10cmin, _log10cmax = 0.3010299956639812, 2.3010299956639813
        _n = 20
        _t_coords = np.linspace(_t_ellapsed_min, _t_ellapsed_max, _n)
        _chost_coords = np.linspace(_chostmin, _chostmax, _n)
        _log10c_coords = np.linspace(_log10cmin, _log10cmax, _n)
        _values = np.array(_log10_mbound_over_minfall).reshape(_n, _n, _n)
        _values_scatter = np.array(_log10_mbound_over_minfall_scatter_dex).reshape(_n, _n, _n)
        _points = (_t_coords, _chost_coords, _log10c_coords)
        self._mfrac_interp = RegularGridInterpolator(_points, _values, bounds_error=False, fill_value=None)
        self._mfrac_scatter_dex_interp = RegularGridInterpolator(_points, _values_scatter, bounds_error=False,
                                                                 fill_value=None)
        self._afit_interp = RegularGridInterpolator(_points, a_JohnsonSU.reshape(_n, _n, _n),
                                                    bounds_error=False, fill_value=None
                                                    )
        self._bfit_interp = RegularGridInterpolator(_points, b_JohnsonSU.reshape(_n, _n, _n),
                                                    bounds_error=False, fill_value=None
                                        )

    def evaluate_mean_mass_loss(self, log10_concentration_infall, time_since_infall, host_concentration):
        """

        :param log10_concentration_infall: log10(c) where c is the halo concentration at infall
        :param time_since_infall: the time ellapsed since infall and the deflector redshift
        :param host_concentration: the concentration of the host halo
        :return: bound mass divided by the infall mass
        """
        point = (time_since_infall, host_concentration, log10_concentration_infall)
        y = self._mfrac_interp(point)
        if isinstance(log10_concentration_infall, float) and \
            isinstance(time_since_infall, float) and \
            isinstance(host_concentration, float):
            return float(y)
        else:
            return np.squeeze(y)

    def evaluate_scatter_dex(self, log10_concentration_infall, time_since_infall, host_concentration):
        """

        :param log10_concentration_infall: log10(c) where c is the halo concentration at infall
        :param time_since_infall: the time ellapsed since infall and the deflector redshift
        :param host_concentration: the concentration of the host halo
        :return: the scatter in dex of the bound mass divided by infall mass
        """
        point = (time_since_infall, host_concentration, log10_concentration_infall)
        y = self._mfrac_scatter_dex_interp(point)
        if isinstance(log10_concentration_infall, float) and \
            isinstance(time_since_infall, float) and \
            isinstance(host_concentration, float):
            return float(y)
        else:
            return np.squeeze(y)

    def evaluate_JohnsonSU(self, log10_concentration_infall, time_since_infall, host_concentration):
        """

        :param log10_concentration_infall: log10(c) where c is the halo concentration at infall
        :param time_since_infall: the time ellapsed since infall and the deflector redshift
        :param host_concentration: the concentration of the host halo
        :return: the scatter in dex of the bound mass divided by infall mass
        """
        point = (time_since_infall, host_concentration, log10_concentration_infall)
        a = self._afit_interp(point)
        b = self._bfit_interp(point)
        return abs(float(a)), abs(float(b))

    def __call__(self, log10_concentration_infall, time_since_infall, host_concentration, use_JohnsonSU=True):
        """
        Evaluates the prediction from galacticus for subhalo bound mass
        :param log10_concentration_infall: log10(c) where c is the halo concentration at infall
        :param time_since_infall: the time ellapsed since infall and the deflector redshift
        :param host_concentration: the concentration of the host halo
        :return: the log10(bound mass divided by the infall mass), plus scatter
        """
        if use_JohnsonSU:
            if isinstance(log10_concentration_infall, list) or isinstance(log10_concentration_infall, np.ndarray):
                raise Exception('the Johnson SU sampling method only works for one value of concentration, time, and '
                                'host concentration at a time')
            if isinstance(time_since_infall, list) or isinstance(time_since_infall, np.ndarray):
                raise Exception('the Johnson SU sampling method only works for one value of concentration, time, and '
                                'host concentration at a time')
            if isinstance(host_concentration, list) or isinstance(host_concentration, np.ndarray):
                raise Exception('the Johnson SU sampling method only works for one value of concentration, time, and '
                                'host concentration at a time')
            a, b = self.evaluate_JohnsonSU(log10_concentration_infall, time_since_infall, host_concentration)
            try:
                log10_mbound_over_minfall = float(johnsonsu.rvs(a, b))
            except:
                print('failed at point: ', log10_concentration_infall, time_since_infall, host_concentration)
                raise Exception('a, b = '+str(a)+', '+str(b))

        else:
            mean = self.evaluate_mean_mass_loss(log10_concentration_infall, time_since_infall, host_concentration)
            scatter_dex = self.evaluate_scatter_dex(log10_concentration_infall, time_since_infall, host_concentration)
            scatter_dex = max(scatter_dex, 0.001)
            log10_mbound_over_minfall = np.random.normal(mean, scatter_dex)


        if isinstance(log10_concentration_infall, float) and \
            isinstance(time_since_infall, float) and \
            isinstance(host_concentration, float):
            output = min(0.0, max(-4.0, log10_mbound_over_minfall))
        else:
            inds_high = np.where(log10_mbound_over_minfall > 0.0)
            log10_mbound_over_minfall[inds_high] = 0.0
            inds_low = np.where(log10_mbound_over_minfall < -4.0)
            log10_mbound_over_minfall[inds_low] = -4.0
        return output
