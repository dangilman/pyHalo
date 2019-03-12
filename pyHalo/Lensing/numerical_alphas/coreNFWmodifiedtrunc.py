import numpy as np
from scipy.interpolate import interp1d
from pyHalo.Lensing.numerical_alphas import cnfwmodtrunc_deflections
from pyHalo.Lensing.numerical_alphas import cnfwmodtrunc_beta
from pyHalo.Lensing.numerical_alphas import cnfwmodtrunc_logx
from pyHalo.Lensing.numerical_alphas import cnfwmodtrunc_trunc

class InterpCNFWmodtrunc(object):

    def __init__(self, xmin = -4, xmax = 4):

        self.deflections = cnfwmodtrunc_deflections.deflections
        self.beta = cnfwmodtrunc_beta.beta
        log_xnfw = cnfwmodtrunc_logx.logx
        self.tau = cnfwmodtrunc_trunc.trunc

        self.split_beta = []
        self.split_trunc = []

        self._xmin, self._xmax = xmin, xmax
        self._betamin = self.beta[0]
        self._betamax = self.beta[-1]
        self._delta_beta = self.beta[1] - self.beta[0]
        self._tau_min = self.tau[0]
        self._tau_max = self.tau[-1]
        self._delta_tau = self.tau[1] - self.tau[0]

        for i, ti in enumerate(self.tau):

            tau_split = []

            for k, bk in enumerate(self.beta):

                tau_split.append(interp1d(log_xnfw, self.deflections[:,i,k]))

            self.split.append(tau_split)

    def _get_closest_tau(self, tau_value, ind):

        minidx = np.argsort(np.absolute(tau_value - self.tau))[ind]

        return minidx

    def _get_closest_tau_double(self, tau_value):

        minidx = np.argsort(np.absolute(tau_value - self.tau))[0:2]
        dt1 = np.absolute(self.beta[minidx[0]] - tau_value)
        w1 = 1-dt1 * self._delta_tau ** -1
        w2 = 1 - w1

        return w1, w2, minidx

    def _get_closest_beta_double(self, beta_value):

        minidx = np.argsort(np.absolute(beta_value - self.beta))[0:2]

        db1 = np.absolute(self.beta[minidx[0]] - beta_value)
        w1 = 1-db1 * self._delta_beta ** -1
        w2 = 1 - w1

        return w1, w2, minidx

    def _get_closest_beta(self, beta_value, ind):

        minidx = np.argsort(np.absolute(beta_value - self.beta))[ind]

        return minidx

    def _get_closest(self, tau, beta):

        minidx_beta = np.argsort(np.absolute(beta - self.beta))[0:2]
        minidx_tau = np.argsort(np.absolute(tau - self.tau))[0:2]

        bval_1, bval_2 = self.beta[minidx_beta[0]], self.beta[minidx_beta[1]]
        tval_1, tval_2 = self.tau[minidx_tau[0]], self.tau[minidx_tau[1]]

        w1 = 1-np.sqrt((bval_1 - beta) ** 2 * self._delta_beta ** -2 +
                               (tval_1 - tau) ** 2 * self._delta_tau ** -2)
        w2 = 1-w1

        return w1, w2, minidx_tau, minidx_beta

    def __call__(self, x, y, Rs, Rc, Rt, norm, center_x, center_y):

        _x = x - center_x
        _y = y - center_y

        R = np.sqrt(_x**2 + _y**2)
        log_xnfw = np.log10(R * Rs ** -1)

        if isinstance(R, float) or isinstance(R, int):
            if log_xnfw <= self._xmin or log_xnfw >= self._xmax:
                return 0.
        else:
            eps = 1e-5
            low_inds = np.where(log_xnfw<=self._xmin)
            high_inds = np.where(log_xnfw>=self._xmax)
            log_xnfw[low_inds] = self._xmin + eps
            log_xnfw[high_inds] = self._xmax - eps

        beta = Rc * Rs ** -1
        tau = Rt * Rs ** -1

        if beta <= self._betamin or beta >= self._betamax:

            w1, w2, tau_inds = self._get_closest_tau_double(tau)

            beta_ind = self._get_closest_beta(beta, 0)

            func1, func2 = self.split[tau_inds[0]][beta_ind], self.split[tau_inds[1]][beta_ind]

            alpha = w1 * func1(log_xnfw) + w2 * func2(log_xnfw)

        elif tau <= self._tau_min or tau >= self._tau_max:

            w1, w2, beta_inds = self._get_closest_beta_double(beta)

            tau_ind = self._get_closest_tau(tau, 0)

            func1, func2 = self.split[tau_ind][beta_inds[0]], self.split[tau_ind][beta_inds[1]]

            alpha = w1 * func1(log_xnfw) + w2 * func2(log_xnfw)

        else:
            w1, w2, minidx_tau, minidx_beta = self._get_closest(tau, beta)
            func1 = self.split[minidx_tau[0]][minidx_beta[0]]
            func2 = self.split[minidx_tau[1]][minidx_beta[1]]
            alpha = w1 * func1(log_xnfw) + w2 * func2(log_xnfw)

        alpha[low_inds] = 0
        alpha[high_inds] = 0

        return norm * alpha


