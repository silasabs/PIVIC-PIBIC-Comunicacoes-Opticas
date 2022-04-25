import cupy as cp
import numpy as np
import scipy.constants as const
from cupy.random import normal
from cupyx.scipy.fft import fft, fftfreq, ifft
from tqdm.notebook import tqdm

from optic.metrics import signal_power


def edfa(Ei, Fs, G=20, NF=4.5, Fc=193.1e12):
    """
    Simple EDFA model

    :param Ei: input signal field [nparray]
    :param Fs: sampling frequency [Hz][scalar]
    :param G: gain [dB][scalar, default: 20 dB]
    :param NF: EDFA noise figure [dB][scalar, default: 4.5 dB]
    :param Fc: optical center frequency [Hz][scalar, default: 193.1e12 Hz]

    :return: amplified noisy optical signal [nparray]
    """
    assert G > 0, "EDFA gain should be a positive scalar"
    assert NF >= 3, "The minimal EDFA noise figure is 3 dB"

    NF_lin = 10 ** (NF / 10)
    G_lin = 10 ** (G / 10)
    nsp = (G_lin * NF_lin - 1) / (2 * (G_lin - 1))
    N_ase = (G_lin - 1) * nsp * const.h * Fc
    p_noise = N_ase * Fs
    noise = normal(0, np.sqrt(p_noise / 2), Ei.shape) + 1j * normal(
        0, np.sqrt(p_noise / 2), Ei.shape
    )
    noise = cp.array(noise).astype(cp.complex64)
    return Ei * cp.sqrt(G_lin) + noise


def manakovSSF(Ei, Fs, paramCh):
    """
    Manakov model split-step Fourier (symmetric, dual-pol.)

    :param Ei: input signal
    :param Fs: sampling frequency of Ei [Hz]
    :param paramCh: object with physical parameters of the optical channel

    :paramCh.Ltotal: total fiber length [km][default: 400 km]
    :paramCh.Lspan: span length [km][default: 80 km]
    :paramCh.hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :paramCh.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :paramCh.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :paramCh.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :paramCh.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
    :paramCh.amp: 'edfa', 'ideal', or 'None. [default:'edfa']
    :paramCh.NF: edfa noise figure [dB] [default: 4.5 dB]

    :return Ech: propagated signal
    """
    # check input parameters
    paramCh.Ltotal = getattr(paramCh, "Ltotal", 400)
    paramCh.Lspan = getattr(paramCh, "Lspan", 80)
    paramCh.hz = getattr(paramCh, "hz", 0.5)
    paramCh.alpha = getattr(paramCh, "alpha", 0.2)
    paramCh.D = getattr(paramCh, "D", 16)
    paramCh.gamma = getattr(paramCh, "gamma", 1.3)
    paramCh.Fc = getattr(paramCh, "Fc", 193.1e12)
    paramCh.amp = getattr(paramCh, "amp", "edfa")
    paramCh.NF = getattr(paramCh, "NF", 4.5)

    Ltotal = paramCh.Ltotal
    Lspan = paramCh.Lspan
    hz = paramCh.hz
    alpha = paramCh.alpha
    D = paramCh.D
    gamma = paramCh.gamma
    Fc = paramCh.Fc
    amp = paramCh.amp
    NF = paramCh.NF

    # fft in CuPy uses only complex64
    prec = cp.complex64

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    # channel parameters
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ ** 2) / (2 * np.pi * c_kms)
    γ = gamma

    c_kms = cp.asarray(c_kms, dtype=prec)  # speed of light (vacuum) in km/s
    λ = cp.asarray(λ, dtype=prec)
    α = cp.asarray(α, dtype=prec)
    β2 = cp.asarray(β2, dtype=prec)
    γ = cp.asarray(γ, dtype=prec)
    hz = cp.asarray(hz, dtype=prec)
    Ltotal = cp.asarray(Ltotal, dtype=prec)

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft).astype(prec)

    Ei_ = cp.asarray(Ei).astype(prec)

    Ech_x = Ei_[:, 0::2].T
    Ech_y = Ei_[:, 1::2].T

    # define linear operator
    linOperator = cp.array(
        cp.exp(-(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω ** 2) * (hz / 2))
    ).astype(prec)

    if Ech_x.shape[0] > 1:
        linOperator = cp.tile(linOperator, (Ech_x.shape[0], 1))
    else:
        linOperator = linOperator.reshape(1, -1)

    for spanN in tqdm(range(1, Nspans + 1)):
        Ech_x = fft(Ech_x)  # polarization x field
        Ech_y = fft(Ech_y)  # polarization y field

        # fiber propagation step
        for stepN in range(1, Nsteps + 1):
            # First linear step (frequency domain)
            Ech_x = Ech_x * linOperator
            Ech_y = Ech_y * linOperator

            # Nonlinear step (time domain)
            Ex = ifft(Ech_x)
            Ey = ifft(Ech_y)
            Ech_x = Ex * cp.exp(
                1j * (8 / 9) * γ * (Ex * cp.conj(Ex) + Ey * cp.conj(Ey)) * hz
            )
            Ech_y = Ey * cp.exp(
                1j * (8 / 9) * γ * (Ex * cp.conj(Ex) + Ey * cp.conj(Ey)) * hz
            )

            # Second linear step (frequency domain)
            Ech_x = fft(Ech_x)
            Ech_y = fft(Ech_y)

            Ech_x = Ech_x * linOperator
            Ech_y = Ech_y * linOperator

        # amplification step
        Ech_x = ifft(Ech_x)
        Ech_y = ifft(Ech_y)

        if amp == "edfa":
            Ech_x = edfa(Ech_x, Fs, alpha * Lspan, NF, Fc)
            Ech_y = edfa(Ech_y, Fs, alpha * Lspan, NF, Fc)
        elif amp == "ideal":
            Ech_x = Ech_x * cp.exp(α / 2 * Nsteps * hz)
            Ech_y = Ech_y * cp.exp(α / 2 * Nsteps * hz)
        elif amp is None:
            Ech_x = Ech_x * cp.exp(0)
            Ech_y = Ech_y * cp.exp(0)

    Ech_x = cp.asnumpy(Ech_x)
    Ech_y = cp.asnumpy(Ech_y)

    Ech = Ei.copy()
    Ech[:, 0::2] = Ech_x.T
    Ech[:, 1::2] = Ech_y.T

    # np.array([Ech_x.reshape(len(Ei),),
    #                 Ech_y.reshape(len(Ei),)]).T

    # Ech = np.array([Ech_x.reshape(len(Ei),),
    #                 Ech_y.reshape(len(Ei),)]).T

    return Ech, paramCh


def setPowerforParSSFM(sig, powers):

    powers_lin = 10 ** (powers / 10) * 1e-3
    powers_lin = powers_lin.repeat(2) / 2

    for i in np.arange(0, sig.shape[1], 2):
        for k in range(0, 2):
            sig[:, i + k] = (
                np.sqrt(powers_lin[i] / signal_power(sig[:, i + k])) * sig[:, i + k]
            )
            print(
                "power mode %d: %.2f dBm"
                % (i + k, 10 * np.log10(signal_power(sig[:, i + k]) / 1e-3))
            )

    return sig
