from pyHalo.Halos.Profiles.nfw import NFW
from pyHalo.Scattering.velocity_averaged import sigma_v_simple, sigma_v_cored_1, sigma_v_cored_2
import numpy as np
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt

def rc_from_M_setsigmav(M200, z, sigma_0, alpha, logmhm=0, cross_type = 'simple'):

    rms_velocity_dispersion = 30*(M200 * 10**-10) ** (1./3)

    sigma_averaged = sigma_0 * (rms_velocity_dispersion * 10 ** -1) ** alpha

    prof = NFW(z_lens=z, z_source=4)
    c = prof.NFW_concentration(M200, z, scatter=False, logmhm=logmhm, c_scale=60, c_power=-0.17)
    rho0, rs, _ = prof.NFW_params_physical(M200, c, z)

    lam, rc = compute_rc(sigma_averaged, rho0, rs)
    return lam, rc

def rc_from_M(M200, z, sigma_0, alpha, logmhm=0, cross_type = 'simple'):

    rms_velocity_dispersion = 30*(M200 * 10**-10) ** (1./3)
    #v = np.sqrt(2./3) * rms_velocity_dispersion
    # from Lisanti 2016 "Lectures on DM Physics"

    sigma_averaged, rho0, rs = compute_sigmav(M200, z, sigma_0, rms_velocity_dispersion,
                                              alpha, logmhm=logmhm, cross_type = cross_type)

    lam, rc = compute_rc(sigma_averaged, rho0, rs)
    return lam, rc

def compute_sigmav(M200, z, cross, v_rms, alpha, logmhm=0, cross_type = 'simple'):

    prof = NFW(z_lens=z, z_source=4)
    c = prof.NFW_concentration(M200, z, scatter=False, logmhm=logmhm, c_scale=60, c_power=-0.17)
    rho0, rs, _ = prof.NFW_params_physical(M200, c, z)

    if cross_type == 'simple':
        return sigma_v_simple(cross, v_rms, -alpha), rho0, rs
    elif cross_type == 'cored_1':
        return sigma_v_cored_1(cross, v_rms, -alpha), rho0, rs
    elif cross_type == 'cored_2':
        return sigma_v_cored_2(cross, v_rms, -alpha), rho0, rs

def compute_rc(sigmav_averaged, rho0, rs):
    """

    :param sigmav_averaged: velocity averaged cross section in units (cm^2/gram) (km/sec)
    :param rho0: density in M_sun / kpc^3
    :param rs: scale radius in kpc
    :return:
    """
    #scattering_rate = 1 is 1 scatter per hubble time
    #cross in units of cm^2/gram km/sec
    # 1 cm^2/gram km/sec = 2.3165e-10 kpc^3 / solar mass / Gyr
    conversion = 2.3165e-10

    k = 0.23 * sigmav_averaged * (rho0 * 10**-8)

    roots = np.roots([1,2,1,-k])
    lam = np.real(np.max(roots[np.where(np.isreal(roots))]))
    return lam, lam*rs

if False:
    alpha = 0.5
    M = np.logspace(6,10, 20)
    s0 = [1, 3, 6]
    logm = np.log10(M)
    col = ['k', 'r', 'g']
    lab = ['$\sigma_0 = 1$', '$\sigma_0 = 3$', '$\sigma_0 = 6$']
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for i, si in enumerate(s0):
        lam, rcore = [], []
        for mi in M:
            l, rc = rc_from_M(mi, 0.5, si, alpha)
            lam.append(l)
            rcore.append(rc)
        ax1.plot(logm, lam, color=col[i], label=lab[i])
        ax2.plot(logm, np.log10(rcore), color=col[i], label = lab[i])
    ax1.legend(fontsize=12)
    ax1.set_xlabel(r'$\log_{10} M$')
    ax1.set_ylabel(r'$\lambda$', rotation = 90, fontsize=16)
    ax2.set_ylabel(r'$r_{\rm{core}}$'+' [kpc]', rotation = 90, fontsize=16)
    ax2.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5])
    ax2.set_xlabel(r'$\log_{10} M$')
    if alpha == 0:
        ax2.annotate('velocity independent\ncross section', xy=(0.1,0.85),
                 xycoords='axes fraction')
    else:
        ax2.annotate(r'$\alpha = $'+str(alpha), xy=(0.1, 0.85),
                     xycoords='axes fraction')
    ax2.set_yticklabels(np.round(10**np.array([-2, -1.5, -1, -0.5, 0, 0.5]),2))
    plt.tight_layout()
    if alpha == 0:
        plt.savefig('v_independent.pdf')
    else:
        plt.savefig('alpha'+str(alpha)+'.pdf')
    plt.show()

if True:
    al = [0, 1, 2]
    M = np.logspace(6, 10, 20)
    s0 = [1, 3, 6]
    units = r'$\frac{\rm{cm^2}}{\rm{g}}$'
    linesty = ['-','--',':']
    logm = np.log10(M)
    col = ['k', 'r', 'g']
    lab_s0 = [r'$\sigma_0 = 1$', r'$\sigma_0 = 3$', r'$\sigma_0 = 6$']

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for k, alpha in enumerate(al):
        for i, si in enumerate(s0):
            lam, rcore = [], []
            for mi in M:
                l, rc = rc_from_M(mi, 0.5, si, alpha, cross_type = 'simple')
                lam.append(l)
                rcore.append(rc)
            if k == 0:
                ax1.plot(logm, lam, color=col[i], label = lab_s0[i]+units, linestyle = linesty[k])
                ax2.plot(logm, np.log10(rcore), color=col[i], label = lab_s0[i], linestyle = linesty[k])
            else:
                ax1.plot(logm, lam, color=col[i], linestyle=linesty[k])
                ax2.plot(logm, np.log10(rcore), color=col[i], linestyle=linesty[k])
    leg1 = ax1.legend(fontsize=12)
    ax1.set_xlabel(r'$\log_{10} M$')
    ax1.set_ylabel(r'$\lambda$', rotation = 90, fontsize=16)
    ax2.set_ylabel(r'$r_{\rm{core}}$'+' [kpc]', rotation = 90, fontsize=16)
    ax2.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5])
    ax2.set_xlabel(r'$\log_{10} M$')

    ax2.set_yticklabels(np.round(10**np.array([-2, -1.5, -1, -0.5, 0, 0.5]),2))

    legend_elements = [Line2D([0], [0], color='k', lw=3, linestyle='-', label=r'$k = 0$'),
                       Line2D([0], [0], color='k', lw=3, linestyle='--', label=r'$k = 1$'),
                       Line2D([0], [0], color='k', lw=3, linestyle=':', label=r'$k = 2$')]
    leg = ax2.legend(handles=legend_elements, loc=2, fontsize=10, frameon=True)
    ax2.annotate(r'$\langle \sigma_v \rangle = \frac{4 \pi}{\left(\pi v_p^2\right)^{\frac{3}{2}}}  \int v^3 \sigma \left(v\right) \rm{exp}\left(-\frac{v^2}{v_p^2}\right) dv$', xy=(0.1,0.04),
                 xycoords='axes fraction', fontsize=9)
    ax2.annotate(r'$\sigma \left(v\right) = \sigma_0 \left(\frac{v}{10 \frac{\rm{km}}{\rm{sec}}}\right)^{-k}$',xy=(0.24,0.15),
                 xycoords='axes fraction', fontsize=11)
    #ax2.add_artist(leg)
    plt.tight_layout()
    plt.savefig('alpha_compare.pdf')
    plt.show()

if False:
    al = [0, 0.5, 1]
    M = np.logspace(6, 10, 20)
    s0 = [5, 10, 20]
    units = r'$\frac{\rm{cm^2}}{\rm{g}} \frac{\rm{km}}{\rm{sec}}$'
    linesty = ['-','--',':']
    logm = np.log10(M)
    col = ['k', 'r', 'g']
    lab_s0 = ['$\sigma_0 = 5$', '$\sigma_0 = 10$', '$\sigma_0 = 20$']

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for k, alpha in enumerate(al):
        for i, si in enumerate(s0):
            lam, rcore = [], []
            for mi in M:
                l, rc = rc_from_M_setsigmav(mi, 0.5, si, alpha, cross_type = 'simple')
                lam.append(l)
                rcore.append(rc)
            if k == 0:
                ax1.plot(logm, lam, color=col[i], label = lab_s0[i]+units, linestyle = linesty[k])
                ax2.plot(logm, np.log10(rcore), color=col[i], label = lab_s0[i], linestyle = linesty[k])
            else:
                ax1.plot(logm, lam, color=col[i], linestyle=linesty[k])
                ax2.plot(logm, np.log10(rcore), color=col[i], linestyle=linesty[k])
    leg1 = ax1.legend(fontsize=12)
    ax1.set_xlabel(r'$\log_{10} M$')
    ax1.set_ylabel(r'$\lambda$', rotation = 90, fontsize=16)
    ax2.set_ylabel(r'$r_{\rm{core}}$'+' [kpc]', rotation = 90, fontsize=16)
    ax2.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5])
    ax2.set_xlabel(r'$\log_{10} M$')

    ax2.set_yticklabels(np.round(10**np.array([-2, -1.5, -1, -0.5, 0, 0.5]),2))

    legend_elements = [Line2D([0], [0], color='k', lw=3, linestyle='-', label=r'$\alpha = 0$'),
                       Line2D([0], [0], color='k', lw=3, linestyle='--', label=r'$\alpha = 0.5$'),
                       Line2D([0], [0], color='k', lw=3, linestyle=':', label=r'$\alpha = 1$')]
    leg = ax2.legend(handles=legend_elements, loc=2, fontsize=10, frameon=True)
    ax2.annotate(r'$\langle \sigma_v \rangle = \sigma_0 \left(\frac{v_p}{10 \frac{\rm{km}}{\rm{sec}}}\right)^{\alpha}$', xy=(0.3,0.1),
                 xycoords='axes fraction', fontsize=14)
    #ax2.add_artist(leg)
    plt.tight_layout()
    plt.savefig('specify_sigmav'+'.pdf')
    plt.show()


