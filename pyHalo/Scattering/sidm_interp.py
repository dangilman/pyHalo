import numpy as np

def get_interps(vpower):

    if vpower == 0:
        zeta_values = np.array([15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        delta_zeta = 10

        slopes_fit15 = np.array([-0.00873369, 0.04547522, -0.29734756])
        intercept_fit15 = np.array([0.07877708, -0.38898387, 10.5698512])

        slopes_fit20 = np.array([-0.00918773, 0.0488793, -0.29863145])
        intercept_fit20 = np.array([0.07010828, -0.36470976, 10.46369133])

        slopes_fit30 = np.array([-0.00241264, 0.02423532, -0.27256061])
        intercept_fit30 = np.array([4.88893147e-03, -1.17866173e-01, 1.01118786e+01])

        slopes_fit40 = np.array([-0.00290296, 0.02673381, -0.2688431])
        intercept_fit40 = np.array([6.07813845e-03, -1.21838139e-01, 1.00005009e+01])

        slopes_fit50 = np.array([0.00044168, 0.01763467, -0.26034332])
        intercept_fit50 = np.array([-0.02583692, -0.02511449, 9.86354876])

        slopes_fit60 = np.array([0.00093022, 0.01567164, -0.25349179])
        intercept_fit60 = np.array([-3.16140435e-02, 2.30547410e-03, 9.75835143e+00])

        slopes_fit70 = np.array([-0.00082639, 0.02240234, -0.25545913])
        intercept_fit70 = np.array([-0.01940403, -0.04151627, 9.73603133])

        slopes_fit80 = np.array([-2.40575064e-04, 2.22435083e-02, -2.53736741e-01])
        intercept_fit80 = np.array([-0.02435385, -0.03561933, 9.69079706])

        slopes_fit90 = np.array([-1.84122523e-04, 2.23516959e-02, -2.49861931e-01])
        intercept_fit90 = np.array([-0.02648198, -0.0264818, 9.62954199])

        slopes_fit100 = np.array([0.00107284, 0.01941302, -0.24634224])
        intercept_fit100 = np.array([-3.73727259e-02, 1.53168502e-03, 9.57909464e+00])

        slopes_fit110 = np.array([-4.55463887e-05, 2.31114321e-02, -2.46727988e-01])
        intercept_fit110 = np.array([-0.03068813, -0.01717815, 9.5574457])

        slopes_fit120 = np.array([-0.00051066, 0.0256819, -0.24616056])
        intercept_fit120 = np.array([-0.02726755, -0.03379491, 9.53504005])

        slope_polynomial = [slopes_fit15, slopes_fit20, slopes_fit30, slopes_fit40, slopes_fit50, slopes_fit60,
                            slopes_fit70, slopes_fit80, slopes_fit90, slopes_fit100, slopes_fit110, slopes_fit120]

        intercept_polynomial = [intercept_fit15, intercept_fit20, intercept_fit30, intercept_fit40, intercept_fit50,
                                intercept_fit60, intercept_fit70, intercept_fit80, intercept_fit90,
                                intercept_fit100, intercept_fit110, intercept_fit120]

    elif vpower == 0.25:

        zeta_values = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        delta_zeta = 10

        v_power = 0.25

        slopes_fit5 = np.array([0.00265324, 0.0013187, -0.23389509])
        intercept_fit5 = np.array([-0.01416015, -0.0436715, 10.29509811])

        slopes_fit10 = np.array([-0.00244904, 0.01822089, -0.22663368])
        intercept_fit10 = np.array([0.01668281, -0.11724189, 9.95671085])

        slopes_fit20 = np.array([-0.00406356, 0.02438003, -0.22862712])
        intercept_fit20 = np.array([0.01234056, -0.0892347, 9.73510325])

        slopes_fit30 = np.array([0.00379082, -0.00085764, -0.20774859])
        intercept_fit30 = np.array([-0.05609376, 0.13923396, 9.45090133])

        slopes_fit40 = np.array([0.00022811, 0.01055014, -0.21209176])
        intercept_fit40 = np.array([-0.02789103, 0.05653192, 9.41670283])

        slopes_fit50 = np.array([0.00156431, 0.00819971, -0.20830772])
        intercept_fit50 = np.array([-0.04147038, 0.0899329, 9.32832227])

        slopes_fit60 = np.array([0.00185098, 0.008343, -0.20757718])
        intercept_fit60 = np.array([-0.04674078, 0.10276386, 9.2784344])

        slopes_fit70 = np.array([0.00265975, 0.00491357, -0.20274943])
        intercept_fit70 = np.array([-0.05378208, 0.13671694, 9.20575002])

        slopes_fit80 = np.array([0.00276225, 0.00710798, -0.20554342])
        intercept_fit80 = np.array([-0.05615454, 0.12678472, 9.20104849])

        slopes_fit90 = np.array([0.00167941, 0.01078237, -0.20590455])
        intercept_fit90 = np.array([-0.05054303, 0.11153491, 9.17377731])

        slopes_fit100 = np.array([0.00130484, 0.01211627, -0.20549199])
        intercept_fit100 = np.array([-0.04693703, 0.10302725, 9.15061961])

        slopes_fit110 = np.array([0.00151193, 0.01281393, -0.2050846])
        intercept_fit110 = np.array([-0.05048566, 0.10626126, 9.12707519])

        slopes_fit120 = np.array([0.00177043, 0.01211839, -0.20268432])
        intercept_fit120 = np.array([-0.05287078, 0.11606177, 9.09110859])

        slope_polynomial = [slopes_fit5, slopes_fit10, slopes_fit20, slopes_fit30, slopes_fit40, slopes_fit50,
                            slopes_fit60,
                            slopes_fit70, slopes_fit80, slopes_fit90, slopes_fit100, slopes_fit110, slopes_fit120]

        intercept_polynomial = [intercept_fit5, intercept_fit10, intercept_fit20, intercept_fit30, intercept_fit40,
                                intercept_fit50,
                                intercept_fit60, intercept_fit70, intercept_fit80, intercept_fit90,
                                intercept_fit100, intercept_fit110, intercept_fit120]

    elif vpower == 0.4:

        zeta_values = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        delta_zeta = 10

        slopes_fit5 = np.array([0.15961935, -0.6056795, 0.27743768])
        intercept_fit5 = np.array([-1.19772122, 4.54910621, 6.3036186])

        slopes_fit10 = np.array([-0.00066629, 0.01213143, -0.19620701])
        intercept_fit10 = np.array([-1.12017166e-02, -7.26001369e-03, 9.59743358e+00])

        slopes_fit20 = np.array([0.00467656, -0.00875179, -0.17701287])
        intercept_fit20 = np.array([-0.0666846, 0.2160626, 9.23163342])

        slopes_fit30 = np.array([0.00280357, 0.00045004, -0.18216039])
        intercept_fit30 = np.array([-0.05221757, 0.15167468, 9.1749843])

        slopes_fit40 = np.array([0.00178267, 0.00395416, -0.18325958])
        intercept_fit40 = np.array([-0.04618201, 0.13696309, 9.11572964])

        slopes_fit50 = np.array([0.00209424, 0.00252393, -0.18106064])
        intercept_fit50 = np.array([-0.05096114, 0.16131746, 9.04605069])

        slopes_fit60 = np.array([0.00301758, 0.002483, -0.18321019])
        intercept_fit60 = np.array([-0.05972626, 0.16918253, 9.0259463])

        slopes_fit70 = np.array([0.00343449, 0.00115228, -0.18177905])
        intercept_fit70 = np.array([-0.06508904, 0.19271624, 8.97753326])

        slopes_fit80 = np.array([3.78459538e-03, -1.21050722e-04, -1.79476704e-01])
        intercept_fit80 = np.array([-0.06889257, 0.2106829, 8.93042337])

        slopes_fit90 = np.array([0.00279943, 0.00364632, -0.18093018])
        intercept_fit90 = np.array([-0.06191401, 0.1872276, 8.91996702])

        slopes_fit100 = np.array([0.00285508, 0.00400034, -0.18209596])
        intercept_fit100 = np.array([-0.06450729, 0.19490901, 8.90676437])

        slopes_fit110 = np.array([0.00415302, -0.00112172, -0.1762421])
        intercept_fit110 = np.array([-0.07595476, 0.24258089, 8.84230485])

        slopes_fit120 = np.array([0.00282095, 0.00537875, -0.18258776])
        intercept_fit120 = np.array([-0.06649255, 0.19674746, 8.87858429])

        slope_polynomial = [slopes_fit5, slopes_fit10, slopes_fit20, slopes_fit30, slopes_fit40, slopes_fit50,
                            slopes_fit60,
                            slopes_fit70, slopes_fit80, slopes_fit90, slopes_fit100, slopes_fit110, slopes_fit120]

        intercept_polynomial = [intercept_fit5, intercept_fit10, intercept_fit20, intercept_fit30, intercept_fit40,
                                intercept_fit50,
                                intercept_fit60, intercept_fit70, intercept_fit80, intercept_fit90,
                                intercept_fit100, intercept_fit110, intercept_fit120]

    elif vpower == 0.5:

        zeta_values = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        delta_zeta = 10

        slopes_fit5 = np.array([0.0012989, -0.00151646, -0.15982686])
        intercept_fit5 = np.array([-0.01229278, 0.0428837, 9.49031537])

        slopes_fit10 = np.array([-0.00025443, 0.01007318, -0.17423818])
        intercept_fit10 = np.array([-0.0169306, 0.02538956, 9.363078])

        slopes_fit20 = np.array([0.00320873, -0.00437765, -0.16328214])
        intercept_fit20 = np.array([-0.05463304, 0.18431916, 9.08253523])

        slopes_fit30 = np.array([0.00308802, -0.00220246, -0.16433341])
        intercept_fit30 = np.array([-0.05864577, 0.19309639, 8.98506665])

        slopes_fit40 = np.array([0.00321089, -0.00261928, -0.16619568])
        intercept_fit40 = np.array([-0.06121329, 0.20783084, 8.93435083])

        slopes_fit50 = np.array([0.00248163, 0.00078203, -0.16693899])
        intercept_fit50 = np.array([-0.05739456, 0.19123372, 8.89263283])

        slopes_fit60 = np.array([0.00483479, -0.0057663, -0.16375015])
        intercept_fit60 = np.array([-0.07757196, 0.25334904, 8.82731654])

        slopes_fit70 = np.array([0.00465157, -0.00559884, -0.16428474])
        intercept_fit70 = np.array([-0.07582748, 0.2570401, 8.80085341])

        slopes_fit80 = np.array([0.00353611, -0.00084991, -0.16869627])
        intercept_fit80 = np.array([-0.06987584, 0.23252822, 8.80845793])

        slopes_fit90 = np.array([0.00476609, -0.00431403, -0.16640688])
        intercept_fit90 = np.array([-0.08219107, 0.27246728, 8.76596329])

        slopes_fit100 = np.array([0.0046619, -0.00344269, -0.1671239])
        intercept_fit100 = np.array([-0.08314166, 0.27455759, 8.75212681])

        slopes_fit110 = np.array([3.60339922e-03, 1.31360162e-04, -1.69504826e-01])
        intercept_fit110 = np.array([-0.07480173, 0.24838737, 8.7579539])

        slopes_fit120 = np.array([0.00449002, -0.00314914, -0.16767864])
        intercept_fit120 = np.array([-0.084442, 0.28699014, 8.72455021])

        slope_polynomial = [slopes_fit5, slopes_fit10, slopes_fit20, slopes_fit30, slopes_fit40, slopes_fit50,
                            slopes_fit60,
                            slopes_fit70, slopes_fit80, slopes_fit90, slopes_fit100, slopes_fit110, slopes_fit120]

        intercept_polynomial = [intercept_fit5, intercept_fit10, intercept_fit20, intercept_fit30, intercept_fit40,
                                intercept_fit50,
                                intercept_fit60, intercept_fit70, intercept_fit80, intercept_fit90,
                                intercept_fit100, intercept_fit110, intercept_fit120]
    else:
        raise Exception('v_power '+str(vpower)+' not recognized.')

    return slope_polynomial, intercept_polynomial, zeta_values, delta_zeta

def _logrho_Mz(m, z, idx, slope_polynomial, intercept_polynomial):

    slope = slope_polynomial[idx][0] * z ** 2 + slope_polynomial[idx][1] * z + slope_polynomial[idx][2]
    intercept = intercept_polynomial[idx][0] * z ** 2 + intercept_polynomial[idx][1] * z + intercept_polynomial[idx][2]
    # print(slope, intercept)

    return intercept + np.log10(m) * slope

def logrho_Mz_0(m, z, zeta, cmean, c, zeta_values, delta_zeta, slope_poly, intercept_poly):

    if zeta >= 15 and zeta < 20:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * 5 ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    elif zeta > 20 and zeta <= 22.5:
        inds = [1, 2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    elif zeta < zeta_values[0]:
        rho_0 = _logrho_Mz(m, z, 0, slope_poly, intercept_poly)
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
        rho_0 = _logrho_Mz(m, z, nmax, slope_poly, intercept_poly)
        derivative = (-_logrho_Mz(m, z, int(nmax-1)) + rho_0) * delta_zeta ** -1
        logrho_central = (zeta - zeta_values[-1]) * derivative + rho_0

    else:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        #w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    logrho_central = logrho_central + 1.75 * (c - cmean) * cmean ** -1

    return logrho_central

def logrho_Mz_25(m, z, zeta, cmean, c, zeta_values, delta_zeta, slope_poly, intercept_poly):

    if zeta >= 5 and zeta < 10:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * 5 ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    elif zeta >= 10 and zeta < 12.5:
        inds = [1, 2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

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
        rho_0 = _logrho_Mz(m, z, nmax, slope_poly, intercept_poly)
        derivative = (-_logrho_Mz(m, z, int(nmax-1)) + rho_0) * delta_zeta ** -1
        logrho_central = (zeta - zeta_values[-1]) * derivative + rho_0

    else:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        #w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    logrho_central = logrho_central + 1.5 * (c - cmean) * cmean ** -1

    return logrho_central


def logrho_Mz_4(m, z, zeta, cmean, c, zeta_values, delta_zeta, slope_poly, intercept_poly):

    if zeta >= 5 and zeta < 10:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * 5 ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    elif zeta >= 10 and zeta < 12.5:
        inds = [1, 2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    elif zeta < zeta_values[0]:
        rho_0 = _logrho_Mz(m, z, 0, slope_poly, intercept_poly)

        rho_at_0 = 10

        derivative = (rho_0 - rho_at_0) * delta_zeta ** -1

        logrho_central = (zeta - zeta_values[0]) * derivative + rho_0

    elif zeta > 120:
        nmax = int(len(zeta_values))-1
        rho_0 = _logrho_Mz(m, z, nmax, slope_poly, intercept_poly)
        derivative = (-_logrho_Mz(m, z, int(nmax-1)) + rho_0) * delta_zeta ** -1
        logrho_central = (zeta - zeta_values[-1]) * derivative + rho_0

    else:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        #w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    logrho_central = logrho_central + 1.75 * (c - cmean) * cmean ** -1

    return logrho_central

def logrho_Mz_5(m, z, zeta, cmean, c, zeta_values, delta_zeta, slope_poly, intercept_poly):

    if zeta >= 5 and zeta < 10:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * 5 ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    elif zeta >= 10 and zeta < 12.5:
        inds = [1, 2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        # w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    elif zeta < zeta_values[0]:
        rho_0 = _logrho_Mz(m, z, 0, slope_poly, intercept_poly)

        rho_at_0 = 10

        derivative = (rho_0 - rho_at_0) * delta_zeta ** -1

        logrho_central = (zeta - zeta_values[0]) * derivative + rho_0

    elif zeta > 120:
        nmax = int(len(zeta_values))-1
        rho_0 = _logrho_Mz(m, z, nmax, slope_poly, intercept_poly)
        derivative = (-_logrho_Mz(m, z, int(nmax-1), slope_poly, intercept_poly) + rho_0) * delta_zeta ** -1
        logrho_central = (zeta - zeta_values[-1]) * derivative + rho_0

    else:

        inds = np.argsort(np.absolute(zeta_values - zeta))[0:2]

        w1 = np.absolute(1 - np.absolute(zeta - zeta_values[inds[0]]) * delta_zeta ** -1)
        w2 = 1 - w1
        #w1, w2 = 1, 0

        rho1 = w1 * _logrho_Mz(m, z, inds[0], slope_poly, intercept_poly)
        rho2 = w2 * _logrho_Mz(m, z, inds[1], slope_poly, intercept_poly)

        logrho_central = rho1 + rho2

    logrho_central = logrho_central + 1.4 * (c - cmean) * cmean ** -1

    return logrho_central

def logrho(m, z, zeta, cmean, c, vpower):

    slope_polynomial, intercept_polynomial, zeta_values, delta_zeta = get_interps(vpower)

    if vpower == 0:
        return logrho_Mz_0(m, z, zeta, cmean, c, zeta_values, delta_zeta, slope_polynomial, intercept_polynomial)
    elif vpower == 0.25:
        return logrho_Mz_25(m, z, zeta, cmean, c, zeta_values, delta_zeta, slope_polynomial, intercept_polynomial)
    elif vpower == 0.4:
        return logrho_Mz_4(m, z, zeta, cmean, c, zeta_values, delta_zeta, slope_polynomial, intercept_polynomial)
    elif vpower == 0.5:
        return logrho_Mz_5(m, z, zeta, cmean, c, zeta_values, delta_zeta, slope_polynomial, intercept_polynomial)
