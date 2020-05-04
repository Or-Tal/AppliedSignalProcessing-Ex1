from EX1_APSP_UTILS import *
import matplotlib.pyplot as plt

AIRPLANE = "./"



def Q1_sec4():
    # -- a
    alpha, sigma, time, freq, tunit, funit = 0.5, 1, 10, 48, 's', 'khz'
    signal = gen_wss_signal(alpha, sigma, time, freq, tunit, funit).flatten()
    print("Q1 section 4.a:")
    print("ampirical mean = {}".format(np.average(signal)))
    print("ampirical 2nd moment = {}".format(np.average(signal ** 2)))

    # -- b
    beta = np.sqrt(0.5 / np.var(signal))
    print("Q1 section 4.b:")
    print("beta = {} \n"
          "testing: mean((beta*Z_n)^2) = {}".format(beta, np.mean((beta * signal) ** 2)))


def Q1_sec5(playb=False):
    # -- a
    print("Q1 section 5.a")
    alpha, sigma, time, freq, tunit, funit = 0.9, 0.5, 2, 48, 's', 'khz'
    opt_estimators = []
    for i in range(5):
        w = optimal_est(alpha, sigma, i + 1)
        opt_estimators.append(w)
        print("opt filter coefficient : {}".format(w))

    # -- b + c
    signal = gen_wss_signal(alpha, sigma, time, freq, tunit, funit)
    est_signals = []
    for i in range(5):
        w = opt_estimators[i]
        est_signal = predict(signal, w)
        est_signals.append(est_signal)
        if playb:
            err = signal - est_signal
            beta = np.sqrt(0.5 / np.var(signal))
            play_audio(beta * signal, 48000)
            play_audio(beta * err, 48000)

    # -- d
    print("\nQ1 section 5.d")
    MSE_arr = []
    for i in range(5):
        e = np.average((signal - est_signals[i]) ** 2)
        MSE_arr.append(e)
        print("L = {}, err = {}".format(i + 1, e))

    # -- e
    print("\nQ1 section 5.e")
    for i in range(5):
        print("NRdb = {}".format(NRdb(signal, est_signals[i])))


def Q2():
    # section 1
    alpha, sigma, L = 0.9, 0.5, 4
    R, P = gen_R_mat_and_P(alpha, sigma, L)
    eig_vals = np.linalg.eigvals(R)
    print("-- section 1 --\neigenvalues of R are:\n"
          "{}\nlargest eigenvalue = {}".format(eig_vals, np.max(eig_vals)))

    # section 2
    M = [0.001, 0.01, 0.1, 0.2]
    N, K = 100, len(M)
    Cn, W = [], []
    wstar = optimal_est(alpha, sigma, L)
    for i in range(K):
        w, C = steepest_descent(np.zeros(L), M[i], N, alpha, sigma, L, wstar)
        Cn.append(C)
        W.append(w)

    # section 3 + 4
    colors = ['red', 'magenta', 'green', 'black']
    wstar = to_col_vec(wstar)
    wstar_norm_sq = np.matmul(wstar.T, wstar)
    for i in range(K):
        tmp = []
        for j in np.arange(N):
            tmp.append(10 * np.log10((Cn[i][j] / wstar_norm_sq)))
        plt.plot(np.arange(N), tmp, c=colors[i])
    plt.xlabel("iteration num")
    plt.ylabel("10 log_10 scale error norm")
    plt.ylim(top=20)
    plt.grid()
    plt.title("10 log_10 scale error norm as a function of iteration, according to $μ ̃$ value")
    plt.legend(["$μ ̃$={}".format(M[i]) for i in range(K)])
    plt.show()


def Q3():
    alpha, sigma, t = 0.9, 0.5, 2
    L = [1, 2, 4]
    Mu = [0.01, 0.001, 0.0001]
    signal = gen_wss_signal(alpha, sigma, t, 48, 's', 'khz')
    print("section 2:")
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    step = 1000
    for i, l in enumerate(L):
        R, P = gen_R_mat_and_P(alpha, sigma, l)
        print("L = {}, R's largest eigenvalue is ~ {}".format(l, np.round(np.max(np.linalg.eigvals(R)), 4)))

        # find the optimal estimation coefficients
        w_star = optimal_est(alpha, sigma, l)
        for j, m in enumerate(Mu):
            # run LMS for m, l parameters
            coefficients, predictions = LMS(signal, m, l)

            # calculate the wanted errors
            err_filter_coef = calc_err_filter_coef(coefficients, w_star)
            rel_predict_err = NRdb(signal[l::], predictions.flatten())
            mean_line = fit_mean_line(np.asarray(err_filter_coef))

            # plot
            x = np.arange(l, len(signal), step)
            axs[i, j].plot(x, err_filter_coef[::step], c='red')
            axs[i, j].plot(x, mean_line[::step], c='black')
            axs[i, j].grid()
            axs[i, j].set_title("L = {}, $\mu$ = {}, NRdb = {}".format(l, m, np.round(rel_predict_err, 3)))
            axs[i, j].set(xlabel="n (sample num.)",
                          ylabel="relative error in db")
            axs[i, j].legend(["coefficient err", "cumulative err mean line", "cumulative NRdb"])
    plt.show()


def Q4():
    # section 1
    lam, L, alpha, sigma, t, reps, step = 0.99, 2, 0.9, 0.5, 10, 3, 1000

    signal = gen_wss_signal(alpha, sigma, t, 48, 's', 'khz')  # initiate the signal
    Delta = np.linspace(1e-5, 5, reps * 2)
    Delta[-1] = 100
    w_star = optimal_est(alpha, sigma, L)

    fig, axs = plt.subplots(2, reps, figsize=(16, 8))
    print("Q4 section 1")
    i, j = 0, 0
    for d in Delta:
        if j == reps:
            j = 0
            i += 1
        print("calc subsection: {}, {}".format(i, j))
        # get the prediction err and coefficints as a function of iterations
        predictions, coefficients = RLS(signal[:10], L, d, lam)
        err_filter_coef = calc_err_filter_coef(coefficients[:10], w_star)
        mean_line = fit_mean_line(np.asarray(err_filter_coef))

        # plot
        x = np.arange(10)
        axs[i, j].plot(x, err_filter_coef, c='red')
        axs[i, j].plot(x, mean_line, c='black')
        axs[i, j].grid()
        axs[i, j].set_title("$\delta$ = {}, $\lambda$ = {}".format(np.round(d, 5), lam))
        axs[i, j].set(xlabel="n (sample num.)",
                      ylabel="relative error in db")
        axs[i, j].legend(["coefficient err", "cumulative err mean line"])

        j += 1

    plt.show()

    fig, axs = plt.subplots(2, reps, figsize=(16, 8))
    print("Q4 section 1")
    i, j = 0, 0
    for d in Delta:
        if j == reps:
            j = 0
            i += 1
        print("calc subsection: {}, {}".format(i, j))
        # get the prediction err and coefficints as a function of iterations
        predictions, coefficients = RLS(signal, L, d, lam)
        err_filter_coef = calc_err_filter_coef(coefficients, w_star)
        rel_predict_err = NRdb(signal, predictions.flatten())
        mean_line = fit_mean_line(np.asarray(err_filter_coef))

        # plot
        x = np.arange(0, len(signal), step)
        axs[i, j].plot(x, err_filter_coef[::step], c='red')
        axs[i, j].plot(x, mean_line[::step], c='black')
        axs[i, j].grid()
        axs[i, j].set_title("$\delta$ = {}, $\lambda$ = {}, NRdb = {}".format(np.round(d, 5), lam, np.round(rel_predict_err, 3)))
        axs[i, j].set_xlabel("n (sample num.)")
        axs[i, j].set_ylabel("relative error in db")
        axs[i, j].legend(["coefficient err", "cumulative err mean line"])

        j += 1

    plt.show()


def Q5(play=True):
    trivial_est = np.ones(1)




if __name__ == "__main__":
    # Q1_sec4()
    # Q1_sec5(False)
    # Q2()
    Q3()
    Q4()
