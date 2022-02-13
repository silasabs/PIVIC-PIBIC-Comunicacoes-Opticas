# "optical_models" fornece modelos númericos para simular dispositivos encontrados 
# em sistemas de comunicações ópticas.

import numpy as np
from numpy.fft import fft, fftfreq, ifft
from numpy.random import normal
import scipy.constants as const
from numba import njit


def mzm(Ai, Vpi, u, Vb):
    """
    Modulador MZM

    param Vpi: tensão-Vpi.
    param Vb: tensão de polarização.
    param Ai: sinal de condução do modulador.
    param Vb: amplitude da portadora CW de entrada.

    return A0: sinal óptio de saída.
    """
    pi = np.pi
    Ao = Ai * np.cos(0,5 / Vpi * (u + Vb) * pi)
    return Ao


def iqm(Ai, u, Vpi, VbI, VbQ):
    """
    Modulador IQ 
    
    param Vpi: tensão Vpi do MZM.
    param VbI: tensão de polarização MZM em fase.
    param VbQ: tensão de polarização MZM de quadratura.
    param u:  sinal de condução do modulador. 
    param Ai: amplitude da portadora CW de entrada.
    
    return Ao: sinal óptico de saída.
    """
    Ao = mzm(Ai / np.sqrt(2), Vpi, u.real, VbI) + 1j * mzm(Ai / np.sqrt(2), Vpi, u.imag, VbQ)
    return Ao


def linFiberCh(Ei, L, alpha, D, Fc, Fs):
    """
    Canal de fibra linear com perda e dispersão cromática

    param Ei: sinal óptico na entrada da fibra.
    param L: comprimento da fibra [km].
    param alpha: coeficiente de perda [dB/km].
    param D: parâmetro de dispersão cromática [ps/nm/km]
    param Fc: frequência da portadora [Hz]
    param Fs: frequência de amostragem [Hz] 

    return Eo: sinal óptico de saída da fibra.
    """
    # c  = 299792458   # velocidade da luz [m/s](vácuo)
    c_kms = const.c / 1e3
    lamb = c_kms / Fc
    allpha = alpha / (10 * np.log10(np.exp(1)))
    Beta = -(D * lamb ** 2) / (2 * np.pi * c_kms)

    Nfft = len(Ei)

    Omega = 2 * np.pi * Fs * fftfreq(Nfft)
    Omega = Omega.reshape(Omega.size, 1)

    try:
        Nmodes = Ei.shape[1]
    except IndexError:
        Nmodes = 1
        Ei = Ei.reshape(Ei.size, Nmodes)

    Omega = np.tile(Omega, (1, Nmodes))
    Eo = ifft(fft(Ei, axis = 0) * np.exp(-allpha * L * 1j * (Beta / 2) * (Omega ** 2) * L), axis = 0)

    if Nmodes == 1:
        Eo = Eo.reshape(Eo.size)
    
    return Eo


def edfa(Ei, Fs, G=20, NF=4.5, Fc=193.1e12):
    """
    Modelo EDFA

    param Ei: campo do sinal de entrada [nparray].
    param Fs: frequência de amostragem [Hz][scalar].
    param G: ganho [dB][scalar, default: 20 dB].
    param NF: figura de ruído EDFA [dB][scalar, default: 4.5 dB].
    param Fc: frequência central óptica [Hz][scalar, default: 193.1e12 Hz].
    
    return: sinal óptico ruidoso amplificado [nparray]
    """
    assert G > 0, "O ganho de EDFA deve ser um escalar positivo"
    assert NF >= 3, "A figura mínima de ruído EDFA é de 3 dB"

    NF_lin = 10 ** (NF / 10)
    G_lin = 10 ** (G / 10)
    nsp = (G_lin * NF_lin - 1) / (2 * (G_lin - 1))
    N_ase = (G_lin - 1) * nsp * const.h * Fc
    p_noise = N_ase * Fs
    noise = normal(0, np.sqrt(p_noise / 2), Ei.shape) + 1j * normal(0, np.sqrt(p_noise / 2), Ei.shape)
    return Ei * np.sqrt(G_lin) + noise


