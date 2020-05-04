from EX1_APSP_UTILS import *
import matplotlib.pyplot as plt

AIRPLANE = "airplane.wav"
CAFE = "cafe.wav"
CITY = "city.wav"
VACCUM = "vacuumcleaner.wav"
DIR = "./external"


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
    alpha, sigma, t = 0.9, 0.5, 10
    L = [1, 2, 4]
    Mu = [0.01, 0.001, 0.0001]
    signal = gen_wss_signal(alpha, sigma, t, 48, 's', 'khz')
    print("section 2:")
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    step = 1500
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

            play_audio(signal[l:100000 + l], 48000)
            play_audio(signal[l:100000 + l] - predictions[:100000].flatten(), 48000)

    plt.show()


def Q4():
    # section 1
    lam, L, alpha, sigma, t, reps, step = 0.99, 2, 0.9, 0.5, 10, 3, 1200

    signal = gen_wss_signal(alpha, sigma, t, 48, 's', 'khz')  # initiate the signal
    Delta = np.linspace(1e-5, 5, reps * 2)
    Delta[-1] = 100
    w_star = optimal_est(alpha, sigma, L)
    calc_p2 = calc_P2(signal[:5])
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
        predictions, coefficients, P2 = RLS(signal, L, d, lam, True)
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

        # compare predicted and actual P2
        print("delta = {}, predicted P2:\n{}".format(d, P2))
        print("calculated P2:\n{}".format(calc_p2))
        j += 1

    plt.show()


def Q5(play=True):
    trivial_est = np.ones(1)
    sounds_compare = []
    for sound in [CITY, CAFE, AIRPLANE, VACCUM]:
        trivials, RLS_out, LMS_out = [], [], []
        N = 400000
        signal = read_audio_file("{}/{}".format(DIR, sound))[:N]
        trivial_estimated_signal = signal - predict(signal, trivial_est)
        trivials.append(trivial_estimated_signal)
        step = 1500

        fig, axs = plt.subplots(5, 3, figsize=(30, 40))
        for k, L in enumerate([2, 4, 8, 20, 30]):
            line_RLS = []
            for i, lam in enumerate([0.3, 0.6, 0.99]):
                predictions, coefficients = RLS(signal, L, lam=lam)
                audio = signal.flatten() - predictions.flatten()
                line_RLS.append(audio)
                if play:
                    print("playing benchmark")
                    play_audio(trivial_estimated_signal[:N], 48000)
                    print("playing RLS L={} lambda={}".format(L, lam))
                    play_audio(audio[:N], 48000)
                    print("done")
                mse_ = mse(signal[500:], predictions[500:])
                nrdb = NRdb(signal[500:], predictions[500:])
                axs[k, i].plot(np.arange(500, audio.shape[0], step), audio[500::step], c='red')
                axs[k, i].set_title("{}, RLS, L = {}, lam = {}\n"
                                    "NRdb = {}, MSE = {}".format(sound, L, lam, np.round(nrdb, 3), np.round(mse_, 6)), fontsize=22)
                axs[k, i].grid()
            RLS_out.append(line_RLS)
        plt.show()

        fig, axs = plt.subplots(4, 5, figsize=(30, 20))
        for k, L in enumerate(range(1, 5)):
            line_LMS = []
            for i, mu in enumerate([0.1, 0.05, 0.01, 0.005, 0.001]):
                coefficients, predictions = LMS(signal, mu, L)
                audio = signal.flatten()[L:] - predictions.flatten()
                line_LMS.append(audio)
                if play:
                    print("playing benchmark")
                    play_audio(trivial_estimated_signal[:N], 48000)
                    print("playing LMS L={} mu={}".format(L, mu))
                    play_audio(audio[:N], 48000)
                    print("done")
                mse_ = mse(signal[L + 500:], predictions[500:])
                nrdb = NRdb(signal[L + 500:], predictions[500:])
                axs[k, i].plot(np.arange(500, audio.shape[0], step), audio[500::step], c='red')
                axs[k, i].set_title("{}, LMS, L = {}, mu = {}\n"
                                    "NRdb = {}, MSE = {}".format(sound, L, mu, np.round(nrdb, 3), np.round(mse_, 6)), fontsize=22)
                axs[k, i].grid()

            LMS_out.append(line_LMS)
        plt.show()

        sounds_compare.append([trivials, LMS_out, RLS_out])
    return sounds_compare


def Q5_results():
    # after preforming some experiments chose the best values for each plot
    best_LMS_ARGS = [(3, 0.1, 35000), (3, 0.1, 40000), (3, 0.1, 40000), (4, 0.1, 4000)]
    best_RLS_ARGS = [(8, 0.99, 100), (20, 0.99, 100), (8, 0.99, 100), (20, 0.99, 100)]
    M = 10000

    for k, sound in enumerate([CITY, CAFE, AIRPLANE, VACCUM]):
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        signal = read_audio_file("{}/{}".format(DIR, sound))
        IP_signal = instantaneous_power(signal, M)

        # LMS
        L1, mu, conv_idx1 = best_LMS_ARGS[k]
        coefficients, predictions = LMS(signal, mu, L1)
        audio1 = signal.flatten()[L1:] - predictions.flatten()
        nrdb1 = NRdb(signal[conv_idx1 + L1:], predictions[conv_idx1:])
        IP_LMS = instantaneous_power(audio1, M)
        axs[0].plot(np.arange(IP_signal.shape[0]), IP_signal, c="blue")
        axs[0].plot(np.arange(IP_LMS.shape[0]), IP_LMS, c="red")
        axs[0].set_title("{}, RLS,L = {}, $\mu$ = {}, noise reduction = {}[dB]".format(sound, L1, mu, np.round(nrdb1, 3)))
        axs[0].set_xlabel("sample number")
        axs[0].set_ylabel("instantaneous power [db]")
        axs[0].grid()
        axs[0].legend(["original noise power", "prediction error power"])

        # RLS
        L2, lam, conv_idx2 = best_RLS_ARGS[k]
        predictions, coefficients= RLS(signal, L2, 1e-5, lam)
        audio2 = signal.flatten() - predictions.flatten()
        nrdb2 = NRdb(signal[conv_idx2:], predictions[conv_idx2:])
        IP_RLS = instantaneous_power(audio2, M)
        axs[1].plot(np.arange(IP_signal.shape[0]), IP_signal, c="blue")
        axs[1].plot(np.arange(IP_RLS.shape[0]), IP_RLS, c="red")
        axs[1].set_title("{}, RLS,L = {}, $\mu$ = {}, noise reduction = {}[dB]".format(sound, L1, mu, np.round(nrdb2, 3)))
        axs[1].set_xlabel("sample number")
        axs[1].set_ylabel("instantaneous power [db]")
        axs[1].grid()
        axs[1].legend(["original noise power", "prediction error power"])

        plt.show()


def Q6(zvec):
    z = _preprocess_Q6(zvec)
    err = 10 ** 10
    znext = None
    for L in [3, 6, 9, 20]:
        for lam in [0.3, 0.6, 0.99]:
            predictions, coefficients = RLS(z, L, lam=lam)
            N = predictions.shape[0]
            _mse = mse(z[N // 10:], predictions[N // 10:])  # assuming that after 10% RLS will converge
            if _mse < err:
                u = to_col_vec(predictions[-L:])
                w = to_col_vec(coefficients[-1])
                znext = np.matmul(u.T, w).flatten()[0]
                err = _mse
    return znext


def _preprocess_Q6(zvec):
    z = np.asarray(zvec).flatten()
    N = min(48000, z.shape[0])
    return z[z.shape[0] - N: z.shape[0]]


if __name__ == "__main__":
    # Q1_sec4()
    # Q1_sec5(False)
    # Q2()
    # Q3()
    # Q4()
    # sounds = Q5(False)
    Q5_results()
