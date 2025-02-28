import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import johnsonsu

class InterpGalacticusKeeley24(object):
    """
    This class interpolates output from the semi-analytic model galacticus to predict the bound mass of a subhalo
    as a function of its infall mass, concentration, host halo concentration, and the time since infall.
    """
    def __init__(self):
        from pyHalo.Halos.galacticus_truncation.tabulated_mass_loss_Keeley24 import _log10_mbound_over_minfall, \
            _log10_mbound_over_minfall_scatter_dex
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

    def __call__(self, log10_concentration_infall, time_since_infall, host_concentration):
        """
        Evaluates the prediction from galacticus for subhalo bound mass
        :param log10_concentration_infall: log10(c) where c is the halo concentration at infall
        :param time_since_infall: the time ellapsed since infall and the deflector redshift
        :param host_concentration: the concentration of the host halo
        :return: the log10(bound mass divided by the infall mass), plus scatter
        """
        mean = self.evaluate_mean_mass_loss(log10_concentration_infall, time_since_infall, host_concentration)
        scatter_dex = self.evaluate_scatter_dex(log10_concentration_infall, time_since_infall, host_concentration)
        scatter_dex = max(scatter_dex, 0.001)
        output = np.random.normal(mean, scatter_dex)
        return output

    def evaluate_mean_mass_loss(self, log10_concentration_infall, time_since_infall, host_concentration):
        """

        :param log10_concentration_infall: log10(c) where c is the halo concentration at infall
        :param time_since_infall: the time ellapsed since infall and the deflector redshift
        :param host_concentration: the concentration of the host halo
        :return: log10 bound mass divided by the infall mass
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

class InterpGalacticus(object):
    """
    This class interpolates output from the semi-analytic model galacticus to predict the bound mass of a subhalo
    as a function of its infall mass, concentration, host halo concentration, and the time since infall.
    """
    def __init__(self):
        from pyHalo.Halos.galacticus_truncation.johnsonSUparams import a_fit, \
            b_fit
        nstep = 15
        log10c_values = np.linspace(np.log10(2.0), np.log10(384), nstep)
        t_inf_values = np.linspace(0.0, 8.1, nstep)
        chost_values = np.linspace(4.0, 9.0, nstep)
        a_values = np.array(a_fit).reshape(nstep, nstep, nstep)
        b_values = np.array(b_fit).reshape(nstep, nstep, nstep)
        _points = (t_inf_values, log10c_values, chost_values)
        self._a_interp = RegularGridInterpolator(_points, a_values, bounds_error=False,
                                               fill_value=None)
        self._b_interp = RegularGridInterpolator(_points, b_values, bounds_error=False,
                                           fill_value=None)

    def __call__(self, log10_concentration_infall, time_since_infall, chost):
        """
        Evaluates the prediction from galacticus for subhalo bound mass
        :param log10_concentration_infall: log10(c) where c is the halo concentration at infall
        :param time_since_infall: the time ellapsed since infall and the deflector redshift
        :param chost: host halo concentration at z=0.5
        :return: the log10(bound mass divided by the infall mass), plus scatter
        """
        log10_concentration_infall = max(np.log10(2), log10_concentration_infall)
        log10_concentration_infall = min(np.log10(384), log10_concentration_infall)
        time_since_infall = max(0.0, time_since_infall)
        time_since_infall = min(time_since_infall, 8.1)
        chost = max(4.0, chost)
        chost = min(8.0, chost)
        p = (time_since_infall, log10_concentration_infall, chost)
        a, b = self._a_interp(p), self._b_interp(p)
        output = float(johnsonsu.rvs(a, b))
        #output -= 0.25
        return min(output, 0.0)
