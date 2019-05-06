import numpy as np

v_power = 0.4

zeta_values = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
delta_zeta = 10

slopes_fit5 = np.array([ 0.15961935, -0.6056795 ,  0.27743768])
intercept_fit5 = np.array([-1.19772122,  4.54910621,  6.3036186 ])

slopes_fit10 = np.array([-0.00066629,  0.01213143, -0.19620701])
intercept_fit10 = np.array([-1.12017166e-02, -7.26001369e-03,  9.59743358e+00])

slopes_fit20 = np.array([ 0.00467656, -0.00875179, -0.17701287])
intercept_fit20 = np.array([-0.0666846 ,  0.2160626 ,  9.23163342])

slopes_fit30 = np.array([ 0.00280357,  0.00045004, -0.18216039])
intercept_fit30 = np.array([-0.05221757,  0.15167468,  9.1749843 ])

slopes_fit40 = np.array([ 0.00178267,  0.00395416, -0.18325958])
intercept_fit40 = np.array([-0.04618201,  0.13696309,  9.11572964])

slopes_fit50 = np.array([ 0.00209424,  0.00252393, -0.18106064])
intercept_fit50 = np.array([-0.05096114,  0.16131746,  9.04605069])

slopes_fit60 = np.array([ 0.00301758,  0.002483  , -0.18321019])
intercept_fit60 = np.array([-0.05972626,  0.16918253,  9.0259463 ])

slopes_fit70 = np.array([ 0.00343449,  0.00115228, -0.18177905])
intercept_fit70 = np.array([-0.06508904,  0.19271624,  8.97753326])

slopes_fit80 = np.array([ 3.78459538e-03, -1.21050722e-04, -1.79476704e-01])
intercept_fit80 = np.array([-0.06889257,  0.2106829 ,  8.93042337])

slopes_fit90 = np.array([ 0.00279943,  0.00364632, -0.18093018])
intercept_fit90 = np.array([-0.06191401,  0.1872276 ,  8.91996702])

slopes_fit100 = np.array([ 0.00285508,  0.00400034, -0.18209596])
intercept_fit100 = np.array([-0.06450729,  0.19490901,  8.90676437])

slopes_fit110 = np.array([ 0.00415302, -0.00112172, -0.1762421 ])
intercept_fit110 = np.array([-0.07595476,  0.24258089,  8.84230485])

slopes_fit120 = np.array([ 0.00282095,  0.00537875, -0.18258776])
intercept_fit120 = np.array([-0.06649255,  0.19674746,  8.87858429])

slope_polynomial = [slopes_fit5 ,slopes_fit10, slopes_fit20, slopes_fit30, slopes_fit40, slopes_fit50, slopes_fit60,
          slopes_fit70, slopes_fit80, slopes_fit90, slopes_fit100, slopes_fit110, slopes_fit120]

intercept_polynomial = [intercept_fit5, intercept_fit10, intercept_fit20, intercept_fit30, intercept_fit40, intercept_fit50,
              intercept_fit60, intercept_fit70, intercept_fit80, intercept_fit90,
              intercept_fit100, intercept_fit110, intercept_fit120]

def _logrho_Mz(m, z, idx):

    slope = slope_polynomial[idx][0] * z ** 2 + slope_polynomial[idx][1] * z + slope_polynomial[idx][2]
    intercept = intercept_polynomial[idx][0] * z ** 2 + intercept_polynomial[idx][1] * z + intercept_polynomial[idx][2]
    # print(slope, intercept)

    return intercept + np.log10(m) * slope

def logrho_Mz_4(m, z, zeta, cmean, c):

    if zeta >= 5 and zeta < 10:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * 5 ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        rho_central = rho1 + rho2

    elif zeta >= 10 and zeta < 12.5:
        inds = [1, 2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        rho_central = rho1 + rho2

    elif zeta < zeta_values[0]:
        rho_0 = _logrho_Mz(m, z, 0)

        rho_at_0 = 10

        derivative = (rho_0 - rho_at_0) * delta_zeta ** -1

        rho_central = (zeta - zeta_values[0]) * derivative + rho_0

    elif zeta > 120:
        nmax = int(len(zeta_values))-1
        rho_0 = _logrho_Mz(m, z, nmax)
        derivative = (-_logrho_Mz(m, z, int(nmax-1)) + rho_0) * delta_zeta ** -1
        rho_central = (zeta - zeta_values[-1]) * derivative + rho_0

    else:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        #w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0])
        rho2 = w2 * _logrho_Mz(m, z, inds[1])

        rho_central = rho1 + rho2

    delta_c = c * cmean ** -1 - 1
    slope_c = 0.5

    rho_central = 10**rho_central * (1 - slope_c * delta_c)

    return np.log10(rho_central)
