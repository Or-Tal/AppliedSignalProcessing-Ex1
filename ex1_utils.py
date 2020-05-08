import sys
import traceback
import numpy as np
import scipy.signal as ss
import scipy.linalg as sl
import sounddevice as sd
from scipy.io.wavfile import read, write


# --------------- Algorithms ---------------
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
        w = w + mu * (P - np.matmul(R, w))

        # update output err array
        if wstar is not None:
            e = np.matmul((w - wstar).T, (w - wstar))[0][0]
            Cn.append(e)

    if wstar is not None:
        return w, Cn
    return w


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
        Un = np.flip(signal[n - L:n])
        Dn_est = np.matmul(w.T, Un, dtype=np.float64).flatten()[0]

        # compute estimation err
        en = signal[n][0] - Dn_est

        # update weights vector
        w = w + mu * Un * en

        # update output arrays
        coefficients.append(w.flatten())
        predictions.append(Dn_est)

    coef_shape = np.shape(coefficients)
    pred_shape = np.shape(predictions)
    return np.asarray(coefficients).reshape(coef_shape), np.asarray(predictions).reshape(pred_shape)


def RLS(signal: np.ndarray, L, delta=1e-2, lam=0.99, p2_flag=False):
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
    P2 = 0
    for n in np.arange(L, signal.shape[0]):
        # compute the estimated value Dn
        Un = np.flip(signal[n - L:n]).reshape((L, 1))
        Dn_est = np.matmul(w.T, Un, dtype=np.float64).flatten()
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

        # save P2 for the output for comparison of delta
        if n == 2:
            P2 = P
    if p2_flag:
        return np.asarray(estimations).flatten(), coefficients, P2
    return np.asarray(estimations).flatten(), coefficients


# --------------- signal operations ---------------
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


def optimal_est(alpha, sigma, L):
    """calculates the optimal estimator for the given params"""
    # get R matrix and P vector
    R, P = gen_R_mat_and_P(alpha, sigma, L)
    # optimal w = R^-1 @ P
    return np.matmul(np.linalg.inv(R), P).flatten()


def predict(signal: np.ndarray, w: np.ndarray):
    """
    :param signal: input samples
    :param w: coefficient vector
    :return: estimated signal (convolve with input coef vector padded with zero)
    """
    signal, w = signal.flatten(), w.flatten()
    w = np.pad(w, (1, 0))
    prediction = ss.lfilter(w, [1], signal)
    return prediction.flatten()


# --------------- functional calculations ---------------
def mse(z: np.ndarray, z_hat: np.ndarray):
    """calculates mean squared err between ground truth and prediction"""
    assert z.shape[0] == z_hat.shape[0]
    z, z_hat = to_col_vec(z), to_col_vec(z_hat)
    err = np.matmul((z-z_hat).T, (z-z_hat)).flatten()[0]/(z.shape[0])
    return err


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


def NRdb(Z: np.ndarray, Z_hat: np.ndarray):
    """
    calculate the NRdb as specified in ex sheet
    """
    var = np.mean(Z ** 2)
    mse = np.mean((Z - Z_hat) ** 2)
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


def instantaneous_power(signal: np.ndarray, M: int):
    """
    calculates the instantaneous power of the signal
    according to the formula: 1/M * sum^(M-1)_(l=0){signal[n-l] ** 2}
    :param signal: signal to estimate
    :param M: window width
    :return: result according to formula, in db scale
    """
    signal = signal.flatten() ** 2
    filter = np.ones(M)
    err = ss.lfilter(filter, [1], signal) / M
    return 10 * np.log10(err)


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


def calc_P2(u: np.ndarray):
    """generates P2 matrix for comparison"""
    u = to_col_vec(u)
    lam = 0.99
    phi = lam * np.matmul(u[:2], u[:2].T) + np.matmul(u[1:3], u[1:3].T)
    return np.linalg.inv(phi.reshape((2,2)))


# --------------- General utils ---------------
def to_col_vec(vec: np.ndarray):
    """transforms vector to col vector"""
    vec = vec.flatten()
    return vec.reshape((vec.shape[0], 1))


def play_audio(audio: np.ndarray, frec):
    """
    :param audio: array of audio input
    :param frec: sampling frequency in Hz
    :return:
    """
    sd.play(audio, samplerate=frec, blocking=True)


def fit_mean_line(input: np.ndarray):
    """fits an average line for cumulative input"""
    input = input.flatten()
    cum_signal = np.cumsum(input)
    output = cum_signal / np.arange(1, input.shape[0] + 1)
    return output


def read_audio_file(path):
    """reads MONO audio file and normalize it's values"""
    audio = read(path)[1]
    return normalize(audio)


def normalize(arr: np.ndarray):
    m1 = np.max(arr)
    m2 = np.min(arr)
    val = max(m1, abs(m2))
    factor = 1
    while factor < val:
        factor *= 2
    return arr / factor


def preprocess_Q6(zvec, w_size):
    """
    clip the last second of the input vector and cast it to np.ndarray
    :param zvec: input vector to estimate
    :param w_size: window size
    :return: clipped vector ndarray
    """
    factor = 1
    z = np.asarray(zvec, dtype=np.float64).flatten()
    m = max(np.max(z), abs(np.min(z)))
    i = 1
    while m > factor:
        factor = 2 ** i
        i += 1
    N = min(w_size, z.shape[0])
    return z[z.shape[0] - N: z.shape[0]].flatten(), factor


def save_audio_file(path, data):
    return write(path, 48000, data)
