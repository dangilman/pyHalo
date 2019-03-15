import numpy as np
from scipy.interpolate import interp1d
from pyHalo.Lensing.numerical_alphas import cnfwmodtrunc_deflections
from pyHalo.Lensing.numerical_alphas import cnfwmodtrunc_beta
from pyHalo.Lensing.numerical_alphas import cnfwmodtrunc_logx
from pyHalo.Lensing.numerical_alphas import cnfwmodtrunc_trunc

class InterpCNFWmodtrunc(object):

    def __init__(self):

        self.deflections = cnfwmodtrunc_deflections.deflections
        self.beta = cnfwmodtrunc_beta.beta
        log_xnfw = cnfwmodtrunc_logx.logx
        self.tau = cnfwmodtrunc_trunc.trunc

        self.split = []

        self._xmin, self._xmax = log_xnfw[0], log_xnfw[-1]
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

    def __call__(self, x, y, Rs, r_core, r_trunc, norm):

        R = np.sqrt(x**2 + y**2)
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

        beta = r_core * Rs ** -1
        tau = r_trunc * Rs ** -1

        tmin = self._get_closest_tau(tau, 0)
        #tmin_2 = self._get_closest_tau(tau, 1)
        bmin = self._get_closest_beta(beta, 0)

        func = self.split[tmin][bmin]

        return norm * func(log_xnfw)

    def old__call__(self, x, y, Rs, Rc, Rt, norm):

        R = np.sqrt(x**2 + y**2)
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

if False:
    c = InterpCNFWmodtrunc()
    x = np.logspace(-1, 1.6, 100)
    y = 0
    Rs = 1
    Rc = 0.5
    Rt = 30
    norm  =1
    import matplotlib.pyplot as plt
    alpha = c(x, y ,Rs, 0.001*Rs, Rt, norm)
    plt.plot(np.log10(x), alpha, color='k', label='NFW')
    alpha = c(x, y ,Rs, 0.2*Rs, Rt, norm)
    plt.plot(np.log10(x), alpha, color='g', label = r'$r_c = 0.2 r_s$')
    alpha = c(x, y ,Rs, 0.5*Rs, Rt, norm)
    plt.plot(np.log10(x), alpha, color='r', label = r'$r_c = 0.5 r_s$')
    alpha = c(x, y ,Rs, 0.25*Rs, 5*Rs, norm)
    plt.plot(np.log10(x), alpha, color='m',label = r'$r_c = 0.25 r_s$'+'\n'+r'$r_t = 5 r_s$')
    plt.legend(fontsize=14, loc=1)
    plt.xlabel(r'$\log_{10} \left(x \right)$',fontsize=14)
    plt.ylabel('deflection angle',fontsize=14)
    text = r'$\rho \left(r, r_c, r_s, r_t\right) = \frac{\rho_0}{\left(r^{10} + r_c^{10}\right)^{\frac{1}{10}} \ \left(r+r_s\right)^2} \ \frac{r_t^2}{r^2+r_t^2}$'
    plt.annotate(text, xy=(0.14, 0.1), xycoords='axes fraction', fontsize=16)
    plt.tight_layout()
    plt.savefig('cored_truncated_nfw.pdf')
    plt.show()

if False:
    ndef = 1501
    ntau = 30
    nbeta = 252
    #ndef = 10
    #ntau = 5
    #nbeta = 3
    deftable = np.zeros((ndef, ntau, nbeta))

    def unpack(beta, tau):
        if beta == 0.001:
            bval = 0.0
        else:
            bval = beta
        deflection, logx = np.loadtxt('tnfwmod_table/table_'+str(bval)+'_'+str(tau)+'.txt', unpack=True)
        return np.round(deflection,6), logx

    betamin, betamax = 0.001, 5.001
    taumin, taumax = 1, 30
    tau_values = np.round(np.linspace(taumin, taumax, 30), 1)
    beta_values = np.append(0.001,np.round(np.arange(betamin+0.02, betamax+0.02, 0.02),2))

    print('building deftable... ')
    for i,ti in enumerate(tau_values):
        print(str(i+1)+' of '+str(len(tau_values)))
        for j,bi in enumerate(beta_values):

            defi, logx = unpack(bi, ti)
            deftable[:,i,j] = defi

    with open('cnfwmodtrunc_logx.py','w') as f:
        f.write('import numpy as np\n')
        f.write('logx = np.array([')
        for bi in logx:
            f.write(str(bi)+', ')
        f.write('])')

    with open('cnfwmodtrunc_beta.py','w') as f:
        f.write('import numpy as np\n')
        f.write('beta = np.array([')
        for bi in beta_values:
            f.write(str(bi)+', ')
        f.write('])')

    with open('cnfwmodtrunc_trunc.py','w') as f:
        f.write('import numpy as np\n')
        f.write('trunc = np.array([')
        for bi in tau_values:
            f.write(str(bi)+', ')
        f.write('])')

    with open('cnfwmodtrunc_deflections.py', 'w') as f:
        f.write('import numpy as np\n')
        f.write('deflections = np.array([')
        for i in range(0, ndef):
            f.write('[')
            for k in range(0, ntau):
                f.write('[')
                for j in range(0, nbeta):
                    f.write(str(deftable[i,k,j])+', ')
                f.write('],\n')
            if i < ndef-1:
                f.write('],\n\n')
            else:
                f.write(']')
        f.write('])')





