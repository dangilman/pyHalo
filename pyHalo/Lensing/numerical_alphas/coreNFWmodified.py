import numpy as np
from scipy.interpolate import interp1d
from pyHalo.Lensing.numerical_alphas import cnfwmod_deflections
from pyHalo.Lensing.numerical_alphas import cnfwmod_beta
from pyHalo.Lensing.numerical_alphas import cnfwmod_logx

class InterpCNFWmod(object):

    def __init__(self, xmin = -3, xmax = 3):

        self.deflections = cnfwmod_deflections.deflections
        self.beta = cnfwmod_beta.beta
        log_xnfw = cnfwmod_logx.logx

        self.split = []
        self._xmin, self._xmax = xmin, xmax
        self._betamin = self.beta[0]
        self._betamax = self.beta[-1]

        assert np.shape(self.deflections)[1] == len(self.beta)

        for i, bi in enumerate(self.beta):

            self.split.append(interp1d(log_xnfw, self.deflections[:,i]))

    def _get_interp_function(self, beta_value):

        minidx = np.argmin(np.absolute(beta_value - self.beta))

        return self.split[minidx]

    def _get_interp_function_avg(self, beta_value):

        minidx = np.argsort(np.absolute(beta_value - self.beta))[0:2]

        r1 = np.absolute(beta_value - self.beta[minidx[0]])

        delta_beta = self.beta[1] - self.beta[0]
        w1 = 1-r1 * delta_beta ** -1
        w2 = 1 - w1

        return w1, self.split[minidx[0]], w2, self.split[minidx[1]]

    def __call__(self, x, y, Rs, r_core, norm):

        R = np.sqrt(x**2 + y**2)
        log_xnfw = np.log10(R * Rs ** -1)

        if isinstance(R, float) or isinstance(R, int):
            if log_xnfw < self._xmin or log_xnfw > self._xmax:
                return 0.

            else:
                func = self._get_interp_function(r_core * Rs ** -1)
                alpha = func(log_xnfw)
                return norm * alpha

        else:
            eps = 1e-5
            low_inds = np.where(log_xnfw<=self._xmin)
            high_inds = np.where(log_xnfw>=self._xmax)
            log_xnfw[low_inds] = self._xmin + eps
            log_xnfw[high_inds] = self._xmax - eps

        beta = r_core * Rs ** -1

        if beta <= self._betamin or beta >= self._betamax:
            func = self._get_interp_function(beta)
            alpha = func(log_xnfw)
        else:
            w1, func1, w2, func2 = self._get_interp_function_avg(beta)
            alpha = w1 * func1(log_xnfw) + w2 * func2(log_xnfw)

        alpha[low_inds] = 0
        alpha[high_inds] = 0

        return norm * alpha








