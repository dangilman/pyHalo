import numpy as np

v_power = 0

zeta_values = np.array([15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
delta_zeta = 10

slopes_fit15 = np.array([-0.00873369,  0.04547522, -0.29734756])
intercept_fit15 = np.array([ 0.07877708, -0.38898387, 10.5698512 ])

slopes_fit20 = np.array([-0.00918773,  0.0488793 , -0.29863145])
intercept_fit20 = np.array([ 0.07010828, -0.36470976, 10.46369133])

slopes_fit30 = np.array([-0.00241264,  0.02423532, -0.27256061])
intercept_fit30 = np.array([ 4.88893147e-03, -1.17866173e-01,  1.01118786e+01])

slopes_fit40 = np.array([-0.00290296,  0.02673381, -0.2688431 ])
intercept_fit40 = np.array([ 6.07813845e-03, -1.21838139e-01,  1.00005009e+01])

slopes_fit50 = np.array([ 0.00044168,  0.01763467, -0.26034332])
intercept_fit50 = np.array([-0.02583692, -0.02511449,  9.86354876])

slopes_fit60 = np.array([ 0.00093022,  0.01567164, -0.25349179])
intercept_fit60 = np.array([-3.16140435e-02,  2.30547410e-03,  9.75835143e+00])

slopes_fit70 = np.array([-0.00082639,  0.02240234, -0.25545913])
intercept_fit70 = np.array([-0.01940403, -0.04151627,  9.73603133])

slopes_fit80 = np.array([-2.40575064e-04,  2.22435083e-02, -2.53736741e-01])
intercept_fit80 = np.array([-0.02435385, -0.03561933,  9.69079706])

slopes_fit90 = np.array([-1.84122523e-04,  2.23516959e-02, -2.49861931e-01])
intercept_fit90 = np.array([-0.02648198, -0.0264818 ,  9.62954199])

slopes_fit100 = np.array([ 0.00107284,  0.01941302, -0.24634224])
intercept_fit100 = np.array([-3.73727259e-02,  1.53168502e-03,  9.57909464e+00])

slopes_fit110 = np.array([-4.55463887e-05,  2.31114321e-02, -2.46727988e-01])
intercept_fit110 = np.array([-0.03068813, -0.01717815,  9.5574457 ])

slopes_fit120 = np.array([-0.00051066,  0.0256819 , -0.24616056])
intercept_fit120 = np.array([-0.02726755, -0.03379491,  9.53504005])

slope_polynomial = [slopes_fit15, slopes_fit20, slopes_fit30, slopes_fit40, slopes_fit50, slopes_fit60,
          slopes_fit70, slopes_fit80, slopes_fit90, slopes_fit100, slopes_fit110, slopes_fit120]

intercept_polynomial = [intercept_fit15, intercept_fit20, intercept_fit30, intercept_fit40, intercept_fit50,
              intercept_fit60, intercept_fit70, intercept_fit80, intercept_fit90,
              intercept_fit100, intercept_fit110, intercept_fit120]

concentration_redshifts = [0.3, 0.6, 1.2, 1.8, 2.5]
concentration_derivatives = np.array([[0.01, 0.03, 0.033],
                                     [0.037, 0.34, 0.038],
                                     [0.036, 0.04, 0.053],
                                     [0.045, 0.061, 0.073],
                                     [0.058, 0.077, 0.097]])
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

    #d_z = 0.005 * (zeta - 3)
    d_z = 0

    return z * z_linear + z_const + d_z

def _logrho_Mz(m, z, idx):

    slope = slope_polynomial[idx][0] * z ** 2 + slope_polynomial[idx][1] * z + slope_polynomial[idx][2]
    intercept = intercept_polynomial[idx][0] * z ** 2 + intercept_polynomial[idx][1] * z + intercept_polynomial[idx][2]
    # print(slope, intercept)

    return intercept + np.log10(m) * slope

def logrho_Mz_0(m, z, zeta, cmean, c):

    if zeta >= 15 and zeta < 20:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * 5 ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        logrho_central = rho1 + rho2

    elif zeta > 20 and zeta <= 22.5:
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

    logrho_central = logrho_central + 1.75 * (c - cmean) * cmean ** -1

    return logrho_central


