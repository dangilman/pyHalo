import numpy as np

zeta_values = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
delta_zeta = 10

v_power = 0.25

slopes_fit5 = np.array([ 0.00265324,  0.0013187 , -0.23389509])
intercept_fit5 = np.array([-0.01416015, -0.0436715 , 10.29509811])

slopes_fit10 = np.array([-0.00244904,  0.01822089, -0.22663368])
intercept_fit10 = np.array([ 0.01668281, -0.11724189,  9.95671085])

slopes_fit20 = np.array([-0.00406356,  0.02438003, -0.22862712])
intercept_fit20 = np.array([ 0.01234056, -0.0892347 ,  9.73510325])

slopes_fit30 = np.array([ 0.00379082, -0.00085764, -0.20774859])
intercept_fit30 = np.array([-0.05609376,  0.13923396,  9.45090133])

slopes_fit40 = np.array([ 0.00022811,  0.01055014, -0.21209176])
intercept_fit40 = np.array([-0.02789103,  0.05653192,  9.41670283])

slopes_fit50 = np.array([ 0.00156431,  0.00819971, -0.20830772])
intercept_fit50 = np.array([-0.04147038,  0.0899329 ,  9.32832227])

slopes_fit60 = np.array([ 0.00185098,  0.008343  , -0.20757718])
intercept_fit60 = np.array([-0.04674078,  0.10276386,  9.2784344 ])

slopes_fit70 = np.array([ 0.00265975,  0.00491357, -0.20274943])
intercept_fit70 = np.array([-0.05378208,  0.13671694,  9.20575002])

slopes_fit80 = np.array([ 0.00276225,  0.00710798, -0.20554342])
intercept_fit80 = np.array([-0.05615454,  0.12678472,  9.20104849])

slopes_fit90 = np.array([ 0.00167941,  0.01078237, -0.20590455])
intercept_fit90 = np.array([-0.05054303,  0.11153491,  9.17377731])

slopes_fit100 = np.array([ 0.00130484,  0.01211627, -0.20549199])
intercept_fit100 = np.array([-0.04693703,  0.10302725,  9.15061961])

slopes_fit110 = np.array([ 0.00151193,  0.01281393, -0.2050846 ])
intercept_fit110 = np.array([-0.05048566,  0.10626126,  9.12707519])

slopes_fit120 = np.array([ 0.00177043,  0.01211839, -0.20268432])
intercept_fit120 = np.array([-0.05287078,  0.11606177,  9.09110859])

slope_polynomial = [slopes_fit5 ,slopes_fit10, slopes_fit20, slopes_fit30, slopes_fit40, slopes_fit50, slopes_fit60,
          slopes_fit70, slopes_fit80, slopes_fit90, slopes_fit100, slopes_fit110, slopes_fit120]

intercept_polynomial = [intercept_fit5, intercept_fit10, intercept_fit20, intercept_fit30, intercept_fit40, intercept_fit50,
              intercept_fit60, intercept_fit70, intercept_fit80, intercept_fit90,
              intercept_fit100, intercept_fit110, intercept_fit120]

concentration_redshifts = [0.3, 0.6, 1.2, 1.8, 2.5]
concentration_derivatives = np.array([[0.02, 0.03, 0.033],
                                     [0.033, 0.034, 0.039],
                                     [0.041, 0.051, 0.066],
                                     [0.056, 0.065, 0.072],
                                     [0.072, 0.084, 0.1]])
cz_interp_m7 = np.polyfit(concentration_redshifts, concentration_derivatives[:,0], 1)
cz_interp_m8 = np.polyfit(concentration_redshifts, concentration_derivatives[:,1], 1)
cz_interp_m9 = np.polyfit(concentration_redshifts, concentration_derivatives[:,2], 1)

logcm_masses = np.array([7,8,9])
linear = [cz_interp_m7[0], cz_interp_m8[0], cz_interp_m9[0]]
const = [cz_interp_m7[1], cz_interp_m8[1], cz_interp_m9[1]]

cm_interp_1 = np.polyfit(logcm_masses, linear, 1)
cm_interp_2 = np.polyfit(logcm_masses, const, 1)

def concentration_derivative(z, logm, zeta):

    z_linear = np.polyval(cm_interp_1, logm)
    z_const = np.polyval(cm_interp_2, logm)

    d_z = 0.005 * (zeta - 3)

    return z * z_linear + z_const + d_z

def _logrho_Mz(m, z, idx):

    slope = slope_polynomial[idx][0] * z ** 2 + slope_polynomial[idx][1] * z + slope_polynomial[idx][2]
    intercept = intercept_polynomial[idx][0] * z ** 2 + intercept_polynomial[idx][1] * z + intercept_polynomial[idx][2]
    # print(slope, intercept)

    return intercept + np.log10(m) * slope

def logrho_Mz_25(m, z, zeta, cmean, c):

    if zeta >= 5 and zeta < 10:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * 5 ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        logrho_central = rho1 + rho2

    elif zeta >= 10 and zeta < 12.5:
        inds = [1, 2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        logrho_central = rho1 + rho2

    elif zeta < zeta_values[0]:
        rho_0 = _logrho_Mz(m, z, 0)

        logm = np.log10(m)
        if logm < 6:
            rho_at_0 = 10
        elif logm <= 7:
            rho_at_0 = 9
        elif logm <= 8:
            rho_at_0 = 8.8
        else:
            rho_at_0 = 8.5

        derivative = (rho_0 - rho_at_0) * delta_zeta ** -1

        logrho_central = (zeta - zeta_values[0]) * derivative + rho_0

    elif zeta > 120:
        nmax = int(len(zeta_values))-1
        rho_0 = _logrho_Mz(m, z, nmax)
        derivative = (-_logrho_Mz(m, z, int(nmax-1)) + rho_0) * delta_zeta ** -1
        logrho_central = (zeta - zeta_values[-1]) * derivative + rho_0

    else:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        #w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        logrho_central = rho1 + rho2

    logrho_central = logrho_central + 1.5 * (c - cmean) * cmean ** -1

    return logrho_central
