import sys
import traceback
import numpy as np
import scipy.signal as ss
import scipy.linalg as sl
import sounddevice as sd


def predict(signal: np.ndarray, w: np.ndarray):
    signal, w = signal.flatten(), w.flatten()
    w = np.pad(w, (1, 0))
    prediction = ss.lfilter(w, [1], signal)
    return prediction.flatten()


def play_audio(audio: np.ndarray, frec):
    """
    :param audio: array of audio input
    :param frec: sampling frequency in Hz
    :return:
    """
    sd.play(audio, samplerate=frec, blocking=True)


def calc_num_sample_for_t_in_freq(t, freq, tunit='s', funit='hz'):
    """
    calculates how many samples should be in t (seconds) at frequency freq (unit)
    :param t: num of time units
    :param freq: num of frequency units
    :param tunit: time unit
    :param funit: freq unit
    :return: num of samples
    """
    assert freq != 0
    funits = {'hz': 0, 'khz': 3, 'mhz': 6, 'ghz': 9}
    tunits = {'s': 0, 'ms': -3, 'micros': -6, 'ns': -9}
    assert type(funit) == str
    assert type(tunit) == str
    funit = funit.lower()
    tunit = tunit.lower()
    assert funit in funits.keys(), "unsupported unit, available units: hz, khz, mhz, ghz"
    assert tunit in tunits.keys(), "unsupported unit, available units: s, ms, micros, ns"
    freq = freq * (10 ** (funits[funit]))
    t = t * (10 ** (tunits[tunit]))
    return int(np.round(t * freq))


def gen_wss_signal(alpha, sigma, t, freq, tunit='s', funit='hz'):
    """
    generate WSS signal corresponding to t (tunit) for audio sampling freq (funit) frequency
    :param a: coef parameter for filtering function
    :param b: res parameter for filtering function
    :param t: num of time units
    :param freq: num of frequency units
    :param tunit: time unit
    :param funit: freq unit
    :return:
    """
    n = calc_num_sample_for_t_in_freq(t, freq, tunit, funit)
    Gn = np.random.normal(0, 1, n)
    X0 = alpha * np.random.normal(0, 1 / (1 - alpha ** 2))
    Gn[0] += X0
    try:
        Xn = ss.lfilter([1], [1, -alpha], Gn)
        Nn = np.random.normal(0, np.sqrt(sigma), n)
        return Xn + Nn
    except ValueError:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)


def gen_R_mat_and_P(alpha, sigma, L):
    """
    :return: generate cross-correlation matrix R, and cross-correlation vector P, both of order L
    """
    assert alpha != 1
    coef = 1 / (1 - alpha ** 2)
    c = np.asarray([alpha ** i for i in range(L)]).flatten() * coef
    c[0] += sigma
    P = coef * np.asarray([alpha ** i for i in range(1, L + 1)]).flatten()
    R = sl.toeplitz(c)
    return R, P


def steepest_descent(w0, mu, N, alpha, sigma, L, wstar=None):
    """
    stages:
    1. calculate auto-correlation matrix R and cross-correlation vector P
    2. for i=1,...,N
        update coefficient: w_(n+1) = w_n + mu(P - R @ w_n)  // where mu = 0.5 * mu_tilda
    :param w0:          initial coefficient vector
    :param mu:    step factor
    :param N:           num of iterations
    :param alpha, sigma: parameters of the signal
    :param L:           order of the estimator
    :return:            estimated w + optional: squared norm err of each iteration
    """
    w = to_col_vec(w0)  # initial coefficient vector
    wstar = to_col_vec(wstar)
    # calculate R, P
    R, P = gen_R_mat_and_P(alpha, sigma, L)
    P = to_col_vec(P)

    Cn = []
    for i in np.arange(N):
        # update coefficient
        w = w + mu * (P - np.matmul(R, w)) / 2

        # update output err array
        if wstar is not None:
            e = np.matmul((w - wstar).T, (w - wstar))[0][0]
            Cn.append(e)

    if wstar is not None:
        return w, Cn
    return w


def optimal_est(alpha, sigma, L):
    """calculates the optimal estimator for the given params"""
    # get R matrix and P vector
    R, P = gen_R_mat_and_P(alpha, sigma, L)
    # optimal w = R^-1 @ P
    return np.matmul(np.linalg.inv(R), P).flatten()


def to_col_vec(vec: np.ndarray):
    """transforms vector to col vector"""
    vec = vec.flatten()
    return vec.reshape((vec.shape[0], 1))


def LMS(signal: np.ndarray, mu, L):
    """
    Least mean squares algorithm
    stages:
    1.  choose initial guess for w
    2.  for each n = 0,1,2,... do:
        a.  Dn_est = w^T @ Un
        b.  Calculate the estimation error: en = Dn - Dn_est
        c.  Update the weights according to: w_(n+1) = w_n + mu * Un * en

    :param signal: np array = vector -> signal to estimate
    :param mu: step size
    :param L: num of coefficients in filter
    :return: predicted signal starting from L'th coordinate, list of all w coefficients during iterations
    """
    assert len(signal.shape) <= 2
    assert mu > 0
    if len(signal.shape) == 1 or signal.shape[1] == 1:
        signal = to_col_vec(signal)

    # initiate the coefficient vector, create output arrays
    w = np.zeros((L, 1))
    coefficients, predictions = [], []

    for n in np.arange(L, signal.shape[0]):
        # construct the estimator
        Un = signal[n - L:n]
        Dn_est = np.matmul(w.T, Un)[0]

        # compute estimation err
        en = signal[n] - Dn_est

        # update weights vector
        w = w + mu * Un * en

        # update output arrays
        coefficients.append(w.flatten())
        predictions.append(Dn_est)

    coef_shape = np.shape(coefficients)
    pred_shape = np.shape(predictions)
    return np.asarray(coefficients).reshape(coef_shape), np.asarray(predictions).reshape(pred_shape)


def RLS(signal: np.ndarray, L, delta, lam):
    """
    implementation of recursive least squares
    :param signal: ground truth signal
    :param L: num of coefficients in filter
    :param delta, lam: the model's parameters
    :return: list of the estimations, and a list of the coefficients in each iteration
    """
    # init output arrays
    estimations, coefficients = [], []

    # initialize P, w
    signal = np.pad(signal, (L, 0))
    P, w = np.eye(L) / delta, np.zeros((L, 1))
    for n in np.arange(L, signal.shape[0]):
        # compute the estimated value Dn
        Un = signal[n - L:n].reshape((L, 1))
        Dn_est = np.matmul(w.T, Un).flatten()
        estimations.append(Dn_est[0])  # add to output array

        # compute the estimation err
        en = (signal[n] - Dn_est)[0]

        # compute kn
        kn = (np.matmul(P, Un) / lam) / (1 + np.matmul(Un.T, np.matmul(P, Un)) / lam)

        # update weight vector
        w = w + kn * en
        coefficients.append(w.flatten())  # add to output array

        # update P matrix
        P = P / lam - np.matmul(kn, np.matmul(Un.T, P)) / lam

    return np.asarray(estimations).flatten(), coefficients


def optimal_est_RLS(signal: np.ndarray, L, lam):
    """

    :param signal:
    :param L:
    :param lam:
    :return:
    """
    if len(signal.shape) == 1:
        signal = signal.reshape((signal.shape[0], 1))
    coefficients = []

    # initiate phy and gn
    phy, gn = lam * np.matmul(signal[:L], signal[:L].T), lam * signal[:L] * signal[L, 0]
    for i in np.arange(L, signal.shape[0]):
        ui = signal[i - L:i, :]
        di = signal[i, 0]

        # update phy, gn
        phy = lam * phy + np.matmul(ui, ui.T)
        gn = lam * gn + ui * di

        # calculate the optimal estimator and add to output
        coefficients.append(np.matmul(np.linalg.inv(phy), gn).flatten())

    return coefficients


def RLS_relative_coef_err(optimal, predicted):
    """
    :param optimal: optimal coefficient per iteration
    :param predicted: predicted coeficient per iteration
    :return: relative coefficient error per iteration
    """
    opt_shape = np.shape(optimal)
    optimal = np.asarray(optimal).reshape(opt_shape)
    predicted = np.asarray(predicted).reshape(opt_shape)  # should have same shape

    Cn_sq = np.linalg.norm(predicted - optimal, axis=1) ** 2
    return 10 * np.log10(Cn_sq / (np.linalg.norm(optimal, axis=1) ** 2))


def RLS_prediction_err(signal: np.ndarray, prediction: np.ndarray, lam):
    """

    :param signal: ground truth signal
    :param prediction: predicted signal
    :param lam: forgetfulness factor
    :return:
    """
    signal, prediction = signal.flatten(), prediction.flatten()
    out_arr = np.ones(signal.shape[0])
    out_arr[0] = (signal[0] - prediction[0]) ** 2
    for i in np.arange(1, signal.shape[0]):
        out_arr[i] = out_arr[i - 1] / lam + (signal[i] - prediction[i]) ** 2
    return out_arr


def cum_NRdb(Z: np.ndarray, Z_hat: np.ndarray):
    out = []
    for i in range(1, len(Z) + 1):
        var = np.average(Z[:i] ** 2)
        mse = np.average((Z[:i] - Z_hat[:i]) ** 2)
        out.append(10 * np.log10(var / mse))
    return out


def NRdb(Z: np.ndarray, Z_hat: np.ndarray):
    """
    calculate the NRdb as specified in ex sheet
    """
    var = np.average(Z ** 2)
    mse = np.average((Z - Z_hat) ** 2)
    return 10 * np.log10(var / mse)


def calc_err_filter_coef(coef_lst, optimal_est: np.ndarray):
    """
    :param coef_lst: list of all predicted coefficients sorted by iteration num (ascending)
    :param optimal_est: optimal estimator for the problemr
    :return: 10 log10(||est_coef[i]-optimal_est||^2) <- err per iteration
    """
    assert type(coef_lst[0]) == np.ndarray
    err_arr = []
    optimal_est = to_col_vec(optimal_est)
    opt_est_norm = np.matmul(optimal_est.T, optimal_est)
    for coef in coef_lst:
        w = to_col_vec(coef)
        Cn = np.matmul((w - optimal_est).T, (w - optimal_est))
        err_arr.append((10 * np.log10(Cn / opt_est_norm))[0][0])
    return err_arr


def fit_mean_line(input: np.ndarray):
    input = input.flatten()
    cum_signal = np.cumsum(input)
    output = cum_signal / np.arange(1, input.shape[0] + 1)
    return output


def read_audio_file(path):