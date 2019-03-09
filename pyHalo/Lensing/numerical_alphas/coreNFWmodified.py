import numpy as np
from scipy.interpolate import interp1d

class InterpCNFWmod(object):

    def __init__(self, xmin = -3, xmax = 3):

        self.deflections = np.loadtxt('cnfwmod_deflections.txt')
        log_xnfw = np.loadtxt('cnfwmod_logx.txt')
        self.beta = np.loadtxt('cnfwmod_beta.txt')

        self.split = []

        self._xmin, self._xmax = xmin, xmax

        assert np.shape(self.deflections)[1] == len(self.beta)

        for i, bi in enumerate(self.beta):

            new_interp = interp1d(log_xnfw, self.deflections[:,i])
            self.split.append(new_interp)

    def _get_interp_function(self, beta_value):

        minidx = np.argmin(np.absolute(beta_value - self.beta))

        return self.split[minidx]

    def get_alpha(self, x, y, Rs, r_core):

        R = np.sqrt(x**2 + y**2)
        log_xnfw = np.log10(R * Rs ** -1)

        if log_xnfw <= self._xmin or log_xnfw >= self._xmax:
            return 0

        beta = r_core * Rs ** -1

        func = self._get_interp_function(beta)

        return func(np.log10(log_xnfw))
