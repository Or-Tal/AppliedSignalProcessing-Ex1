from ex_1_APSP_utils import *
import matplotlib.pyplot as plt


def Q1(play=False, play2=False):
    print("======= Question 1 =======")
    # section 4
    # -- a
    alpha, sigma = 0.5, 1
    signal = gen_wss_signal(alpha, sigma, 10, 48, 's', 'khz')
    print("-- section 4.a. --\ncalculated ampirical mean: {}".format(calc_emp_mean(signal)))
    print("calculated ampirical second moment: {}".format(calc_emp_second_moment(signal)))

    # # -- b
    beta = np.sqrt(0.5 / np.mean(signal * signal))
    print("-- section 4.b. --\nbeta = {}\n"
          "testing: mean((beta*Z)^2) = {}".format(beta, np.mean(beta ** 2 * (signal * signal))))

    # -- c
    if play:
        play_audio(signal, 48000)
        play_audio(beta * signal, 48000)

    # section 5
    # -- a
    alpha, sigma = 0.9, 0.5
    T = 2
    signal = gen_wss_signal(alpha, sigma, T, 48, 's', 'khz')
    w_vecs = []
    print("-- section 5.a. --")
    for L in range(1, 6):
        w = optimal_est(alpha, sigma, L)
        print("coef vec of order: {} is: {}".format(L, w))
        w_vecs.append(w)

    # -- b

    Z_hats = []
    errors = []
    for i in range(5):
        # use filter according to : Z_hat[n] = 0*Z[n] + w0*Z[n-1] + ... + w[L-1]*Z[n-L]
        Z = predict(signal, w_vecs[i])
        errors.append((signal-Z))
        Z_hats.append(Z)
        if play2:
            plt.plot(range(1, 51), Z[-50:], c='red')
            plt.plot(range(1, 51), signal[-50:], c='black')
            plt.title("section 5.b. predicted and ground truth - last 50 samples; L = {}".format(i + 1))
            plt.legend(["prediction", "ground truth"])
            plt.show()

    # -- c
    beta = np.sqrt(0.5 / np.mean(signal ** 2))
    print("-- section 5.c. --\nbeta = {}\n"
          "testing: mean((beta*Z)^2) = {}".format(beta, np.mean(beta ** 2 * (signal ** 2))))
    if play2:
        for i in range(5):
            play_audio(beta * signal, 48000)
            play_audio(beta * errors[i], 48000)
            print("L = {}, mean((beta * err)^2) = {}".format(i + 1, np.mean(beta ** 2 * (errors[i] ** 2))))

    # -- d
    avg_err = []
    print("-- section 5.d. --")
    colors = ['red', 'magenta', 'green', 'blue', 'black']
    for i in range(5):
        e = np.sum(errors[i] ** 2)/errors[i].shape[0]
        avg_err.append(e)
        print("average estimation error = {}, L = {}".format(e, i + 1))
        plt.plot(range(150), (beta * errors[i])[-150:], '--', c=colors[i])
    plt.plot(range(150), (beta * signal)[-150:], c='orange')
    plt.title("section 5.c. last 150 samples from all scaled noises and scaled signal")
    plt.legend(["err, L={}".format(i + 1) for i in range(5)] + ["signal"])
    plt.show()

    # -- e
    print("-- section 5.e. --")
    NRdb = calc_noise_reduction(signal, errors, 5)


def Q2():
    alpha, sigma, L = 0.9, 0.5, 4

    # section 1
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
    wstar_norm_sq = np.linalg.norm(wstar) ** 2
    for i in range(K):
        tmp = []
        for j in np.arange(N):
            tmp.append(10 * np.log10((Cn[i][j] / wstar_norm_sq)))
        plt.plot(np.arange(N), tmp, c=colors[i])
    plt.xlabel("iteration num")
    plt.ylabel("10 log_10 scale error norm")
    plt.title("10 log_10 scale error norm as a function of iteration, according to $μ ̃$ value")
    plt.legend(["$μ ̃$={}".format(M[i]) for i in range(K)])
    plt.show()


def Q3():
    alpha, sigma, t = 0.9, 0.5, 10
    L = [1, 2, 4]
    Mu = [0.01, 0.001, 0.0001]
    signal = gen_wss_signal(alpha, sigma, t, 48, 's', 'khz')
    print("section 2:")
    for l in L:
        R, P = gen_R_mat_and_P(alpha, sigma, l)
        print("L = {}, R's largest eigenvalue is ~ {}".format(l, np.round(np.max(np.linalg.eigvals(R)), 4)))

        # find the optimal estimation coeficcients
        w_star = optimal_est(alpha, sigma, l)
        for i, m in enumerate(Mu):
            # run LMS for m, l parameters
            coefficients, predictions = LMS(signal, m, l)

            # calculate the wanted errors
            step = 1000
            # TODO: debug here
            # rel_predict_err = calc_relative_pred_err(signal[l::step], predictions[::step])
            rel_predict_err = calc_pred_err_rel(signal[l::], predictions)
            err_filter_coef = calc_norm_relative_weight_err(coefficients[::step], w_star[::step])

            # plot
            x = np.arange(l, len(signal), step)
            plt.plot(x, err_filter_coef, c='red')
            # plt.plot(x, rel_predict_err, c='black')
            # plt.title("L = {}, $\mu$ = {}".format(l, m))
            plt.title("L = {}, $\mu$ = {}, relative prediction err {}".format(l, m, rel_predict_err))
            plt.xlabel("n (sample num.); for visibility: step = {}".format(step))
            plt.ylabel("relative error in db")
            # plt.legend(["coefficient err", "prediction err"])
            plt.show()


def Q4():
    # constants
    lam, L, alpha, sigma, t, reps = 0.99, 2, 0.9, 0.5, 10, 10

    signal = gen_wss_signal(alpha, sigma, t, 48, 's', 'khz')  # initiate the signal
    Delta = np.linspace(1e-5, 0.3, reps)
    w_star_arr = optimal_est_RLS(signal, L, lam)
    for d in Delta:
        # get the prediction err and coefficints as a function of iterations
        pred_err, coefficients = RLS(signal[L:], L, d, lam)


if __name__ == "__main__":
    Q1(play2=False)
    # Q2()
    # Q3()
