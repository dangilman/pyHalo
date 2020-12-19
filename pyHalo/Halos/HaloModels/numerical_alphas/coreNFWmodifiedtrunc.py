import numpy as np
from scipy.interpolate import interp1d
from pyHalo.Halos.HaloModels.numerical_alphas import cnfwmodtrunc_deflections
from scipy.interpolate import RegularGridInterpolator
#raise Exception('When using this class, must first unzip the file cnfwmodtrunc_deflections.py.zip to access the '
#                    'numerical deflections stored there.')

class InterpCNFWmodtrunc(object):

    def __init__(self):

        self.deflections = cnfwmodtrunc_deflections.deflections
        #self.beta = cnfwmodtrunc_beta.beta
        #log_xnfw = cnfwmodtrunc_logx.logx
        #self.tau = cnfwmodtrunc_trunc.trunc

        self.beta = np.arange(0.01, 1.11, 0.01)
        self.tau = np.linspace(1, 35, 35)
        self.log_xnfw = np.linspace(-4, 4, 3501)[0:2626]

        self.split = []

        self._xmin, self._xmax = self.log_xnfw[0], self.log_xnfw[-1]
        self._betamin = self.beta[0]
        self._betamax = self.beta[-1]

        self._tau_min = self.tau[0]
        self._tau_max = self.tau[-1]

        points = (self.log_xnfw, self.tau, self.beta)

        self.un_normalized_interp = RegularGridInterpolator(points, self.deflections)

    def __call__(self, x, y, Rs, r_core, r_trunc, norm):

        R = np.sqrt(x ** 2 + y ** 2)
        X = R/Rs
        log_xnfw = np.log10(X)

        beta = r_core * Rs ** -1
        tau = r_trunc * Rs ** -1

        beta = max(self._betamin, beta)
        beta = min(self._betamax, beta)

        tau = max(self._tau_min, tau)
        tau = min(self._tau_max, tau)

        if isinstance(R, float) or isinstance(R, int):

            if log_xnfw <= self._xmin:
                args = (self._xmin, tau, beta)
                return norm * self.un_normalized_interp(args)

            elif log_xnfw > self._xmax:

                args_matching = (self._xmax, tau, beta)
                matching = norm * self.un_normalized_interp(args_matching)/self._tnfw_def(10 ** self._xmax, tau)

                return matching * self._tnfw_def(X, tau)

            else:

                args = (log_xnfw, tau, beta)
                return norm*self.un_normalized_interp(args)

        else:

            log_xnfw[np.where(log_xnfw < self._xmin)] = self._xmin

            high_inds = np.where(log_xnfw > self._xmax)

            valid_range = np.where(log_xnfw <= self._xmax)

            out = np.empty_like(R)

            args_matching = (self._xmax, tau, beta)
            matching = norm * self.un_normalized_interp(args_matching)/self._tnfw_def(10**self._xmax, tau)

            out[high_inds] = matching * self._tnfw_def(X[high_inds], tau)

            args = (log_xnfw[valid_range], tau, beta)
            out[valid_range] = \
                norm * self.un_normalized_interp(args)


            return out

    def _L(self, x, tau):
        """
        Logarithm that appears frequently
        :param x: r/Rs
        :param tau: t/Rs
        :return:
        """

        return np.log(x * (tau + np.sqrt(tau ** 2 + x ** 2)) ** -1)

    def _F(self, x):
        """
        Classic NFW function in terms of arctanh and arctan
        :param x: r/Rs
        :return:
        """
        if isinstance(x, np.ndarray):
            nfwvals = np.ones_like(x)
            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)
            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arctanh((1 - x[inds1] ** 2) ** .5)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arctan((x[inds2] ** 2 - 1) ** .5)
            return nfwvals

        elif isinstance(x, float) or isinstance(x, int):
            if x == 1:
                return 1
            if x < 1:
                return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
            else:
                return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)

    def _tnfw_def(self, x, tau):

        # revert to NFW normalization for now

        factor = tau ** 2 * (tau ** 2 + 1) ** -2 * (
            (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * self._F(x) + tau * np.pi + (tau ** 2 - 1) * np.log(tau) +
            np.sqrt(tau ** 2 + x ** 2) * (-np.pi + self._L(x, tau) * (tau ** 2 - 1) * tau ** -1))

        return factor * x ** -1

class InterpCNFWmodtruncOld(object):

    def __init__(self):

        self.deflections = cnfwmodtrunc_deflections.deflections
        #self.beta = cnfwmodtrunc_beta.beta
        #log_xnfw = cnfwmodtrunc_logx.logx
        #self.tau = cnfwmodtrunc_trunc.trunc

        self.beta = np.arange(0.01, 1.11, 0.01)
        self.tau = np.linspace(1, 35, 35)

        log_xnfw = np.linspace(-4, 4, 3501)[0:2626]

        self.split = []

        self._xmin, self._xmax = log_xnfw[0], log_xnfw[-1]
        self._betamin = self.beta[0]
        self._betamax = self.beta[-1]
        self._delta_beta = self.beta[2] - self.beta[1]

        self._tau_min = self.tau[0]
        self._tau_max = self.tau[-1]
        self._delta_tau = self.tau[1] - self.tau[0]

        for i, ti in enumerate(self.tau):

            tau_split = []

            for k, bk in enumerate(self.beta):

                tau_split.append(interp1d(log_xnfw, self.deflections[:,i,k]))

            self.split.append(tau_split)

    def _L(self, x, tau):
        """
        Logarithm that appears frequently
        :param x: r/Rs
        :param tau: t/Rs
        :return:
        """

        return np.log(x * (tau + np.sqrt(tau ** 2 + x ** 2)) ** -1)

    def _F(self, x):
        """
        Classic NFW function in terms of arctanh and arctan
        :param x: r/Rs
        :return:
        """
        if isinstance(x, np.ndarray):
            nfwvals = np.ones_like(x)
            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)
            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arctanh((1 - x[inds1] ** 2) ** .5)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arctan((x[inds2] ** 2 - 1) ** .5)
            return nfwvals

        elif isinstance(x, float) or isinstance(x, int):
            if x == 1:
                return 1
            if x < 1:
                return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
            else:
                return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)

    def _tnfw_def(self, x, tau):

        # revert to NFW normalization for now

        factor = tau ** 2 * (tau ** 2 + 1) ** -2 * (
            (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * self._F(x) + tau * np.pi + (tau ** 2 - 1) * np.log(tau) +
            np.sqrt(tau ** 2 + x ** 2) * (-np.pi + self._L(x, tau) * (tau ** 2 - 1) * tau ** -1))

        return factor * x ** -1

    def _get_closest_tau(self, tau_value, ind):

        minidx = np.argsort(np.absolute(tau_value - self.tau))[ind]

        return minidx

    def _get_closest_tau_double(self, tau_value):

        minidx = np.argsort(np.absolute(tau_value - self.tau))[0:2]

        return minidx

    def _get_closest_beta_double(self, beta_value):

        minidx = np.argsort(np.absolute(beta_value - self.beta))[0:2]

        return minidx

    def _get_closest_beta(self, beta_value, ind):

        minidx = np.argsort(np.absolute(beta_value - self.beta))[ind]

        return minidx

    def _euclidean(self, xref, yref, xval, yval):

        return ((xref - xval)**2 + (yref - yval)**2)**0.5

    def __call__(self, x, y, Rs, r_core, r_trunc, norm):

        R = np.sqrt(x ** 2 + y ** 2)
        log_xnfw = np.log10(R * Rs ** -1)

        beta = r_core * Rs ** -1
        tau = r_trunc * Rs ** -1

        tmin = self._get_closest_tau(tau, 0)
        bmin = self._get_closest_beta_double(beta)

        if isinstance(R, float) or isinstance(R, int):

            if log_xnfw <= self._xmin:

                defl = self.split[tmin][bmin[0]](self._xmin)
                return norm * defl

            elif log_xnfw > self._xmax:

                defl = self._tnfw_def(10**log_xnfw, tau)
                defl_interp = norm * self.split[tmin][bmin[0]](self._xmax)
                defl_norm = self._tnfw_def(10**self._xmax, tau)

                return defl * (defl_interp / defl_norm)

            else:

                return norm * self.split[tmin][bmin[0]](log_xnfw)

        else:
            #eps = 0

            log_xnfw[np.where(log_xnfw<self._xmin)] = self._xmin
            high_inds = np.where(log_xnfw > self._xmax)
            valid_range = np.where(log_xnfw <= self._xmax)

        deflections = np.empty_like(R)

        deflections[valid_range] = norm * self.split[tmin][bmin[0]](log_xnfw[valid_range])

        defl_interp = norm * self.split[tmin][bmin[0]](self._xmax)
        defl_norm = self._tnfw_def(10 ** self._xmax, tau)

        deflections[high_inds] = self._tnfw_def(10**log_xnfw[high_inds], tau) * (defl_interp / defl_norm)

        return deflections

if False:
    print('ok')
    c = InterpCNFWmodtrunc()
    x = np.logspace(-2, 4, 100)
    y = 0
    Rs = 1
    Rc = 0.5
    Rt = 30
    norm = 8.5
    import matplotlib.pyplot as plt

    alpha = c(x, y ,Rs, 0.001*Rs, Rt, norm)
    plt.plot(np.log10(x), alpha, color='k', label='NFW')
    alpha = c(x, y ,Rs, 0.3*Rs, Rt, norm)
    plt.plot(np.log10(x), alpha, color='g', label = r'$r_c = 0.2 r_s$')
    alpha = c(x, y ,Rs, 0.5*Rs, Rt, norm)
    plt.plot(np.log10(x), alpha, color='r', label = r'$r_c = 0.5 r_s$')
    plt.legend(fontsize=14, loc=1)
    plt.xlabel(r'$\log_{10} \left(x \right)$',fontsize=14)
    plt.ylabel('deflection angle',fontsize=14)
    text = r'$\rho \left(r, r_c, r_s, r_t\right) = \frac{\rho_0}{\left(r^{10} + r_c^{10}\right)^{\frac{1}{10}} \ \left(r+r_s\right)^2} \ \frac{r_t^2}{r^2+r_t^2}$'
    #plt.annotate(text, xy=(0.14, 0.1), xycoords='axes fraction', fontsize=16)
    plt.tight_layout()
    plt.ylim(0, 12)
    #plt.savefig('cored_truncated_nfw.pdf')
    plt.show()

if False:
    print('ok')
    c = InterpCNFWmodtruncOld()
    x = np.logspace(-2, 4, 100)
    y = 0
    Rs = 1
    Rc = 0.5
    Rt = 30
    norm = 8.5
    import matplotlib.pyplot as plt

    alpha = [c(xi, y ,Rs, 0.001*Rs, Rt, norm) for xi in x]
    plt.plot(np.log10(x), alpha, color='k', label='NFW')
    alpha = [c(xi, y ,Rs, 0.3*Rs, Rt, norm) for xi in x]
    plt.plot(np.log10(x), alpha, color='g', label = r'$r_c = 0.2 r_s$')
    alpha = [c(xi, y ,Rs, 0.5*Rs, Rt, norm) for xi in x]
    plt.plot(np.log10(x), alpha, color='r', label = r'$r_c = 0.5 r_s$')
    plt.legend(fontsize=14, loc=1)
    plt.xlabel(r'$\log_{10} \left(x \right)$',fontsize=14)
    plt.ylabel('deflection angle',fontsize=14)
    plt.ylim(0, 12)
    text = r'$\rho \left(r, r_c, r_s, r_t\right) = \frac{\rho_0}{\left(r^{10} + r_c^{10}\right)^{\frac{1}{10}} \ \left(r+r_s\right)^2} \ \frac{r_t^2}{r^2+r_t^2}$'
    #plt.annotate(text, xy=(0.14, 0.1), xycoords='axes fraction', fontsize=16)
    plt.tight_layout()
    #plt.savefig('cored_truncated_nfw.pdf')
    plt.show()




