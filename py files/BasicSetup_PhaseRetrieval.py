# -*- coding: utf-8 -*-
# +
import matplotlib.pyplot as plt
import numpy as np

from commpy.modulation import QAMModem

from optic.dsp import pulseShape, firFilter, decimate, symbolSync
from optic.models import phaseNoise, KramersKronigRx, linFiberCh

from optic.tx import simpleWDMTx
from optic.core import parameters
from optic.equalization import edc, mimoAdaptEqualizer
from optic.carrierRecovery import cpr
from optic.metrics import fastBERcalc, monteCarloGMI, monteCarloMI, signal_power
from optic.plot import pconst

import scipy.constants as const
# -

# ### Simulation of a single polarization optical signal transmission


# +
# Transmitter parameters:
paramTx = parameters()
paramTx.M = 16                 # order of the modulation format
paramTx.Rs = 32e9              # symbol rate [baud]
paramTx.SpS = 4                # samples per symbol
paramTx.Nbits = 400000         # total number of bits per polarization
paramTx.pulse = "rrc"          # pulse shaping filter
paramTx.Ntaps = 1024           # number of pulse shaping filter coefficients
paramTx.alphaRRC = 0.01        # RRC rolloff
paramTx.Pch_dBm = 0            # power of the optical signal [dBm]
paramTx.Nch = 1                # number of WDM channels
paramTx.Fc = 193.1e12          # central frequency of the optical spectrum
paramTx.freqSpac = 37.5e9      # WDM grid spacing

# Optical channel parameters:
Ltotal = 50      # total link distance [km]
alpha = 0        # fiber loss parameter [dB/km]
D  = 16          # fiber dispersion parameter [ps/nm/km]
Fc = paramTx.Fc  # central optical frequency of the WDM spectrum [Hz]

# Receiver parameters:

# local oscillator (LO)
FO = 1.1*paramTx.Rs/2  # frequency offset 
lw = 100e3             # linewidth
ϕ_lo = 0               # initial phase in rad
Plo_dBm = 12           # power in dBm

# General simulation parameters
chIndex = 0  # index of the channel to be demodulated
plotPSD = True
Fs = paramTx.Rs * paramTx.SpS  # simulation sampling rate
# -


# ### Core simulation code

# +
# generate optical signal signal
sigTx, symbTx_, paramTx = simpleWDMTx(paramTx)

# simulate linear signal propagation
sigCh = linFiberCh(sigTx, Ltotal, alpha, D, Fc, Fs)

# plots optical spectrum before and after transmission

# before
plt.figure()
plt.xlim(paramTx.Fc - Fs / 2, paramTx.Fc + Fs / 2)
plt.psd(
    sigTx[:, 0],
    Fs=Fs,
    Fc=paramTx.Fc,
    NFFT=4 * 1024,
    sides="twosided",
    label="optical spectrum - Tx",
)

# after
plt.psd(
    sigCh,
    Fs=Fs,
    Fc=paramTx.Fc,
    NFFT=4 * 1024,
    sides="twosided",
    label="optical spectrum - Rx",
)
plt.legend(loc="lower left")
plt.title("optical spectrum")

# receiver detection and demodulation

Fc = paramTx.Fc
Ts = 1 / Fs
mod = QAMModem(m=paramTx.M)

freqGrid = paramTx.freqGrid
print(
    "Demodulating channel #%d , fc: %.4f THz, λ: %.4f nm\n"
    % (
        chIndex,
        (Fc + freqGrid[chIndex]) / 1e12,
        const.c / (Fc + freqGrid[chIndex]) / 1e-9,
    )
)

symbTx = symbTx_[:, :, chIndex]

Plo = 10 ** (Plo_dBm / 10) * 1e-3  # power in W

print(
    "Local oscillator P: %.2f dBm, lw: %.2f kHz, FO: %.2f MHz"
    % (Plo_dBm, lw / 1e3, FO / 1e6)
)

# generate LO field
π = np.pi
t = np.arange(0, len(sigCh))*Ts
ϕ_pn_lo = phaseNoise(lw, len(sigCh), Ts)

sigLO = np.sqrt(Plo) * np.exp(-1j * (2 * π * FO * t + ϕ_lo + ϕ_pn_lo))

# Add LO to the received signal
sigRx = sigCh + sigLO

print('CSPR = %.2f dB'%(10*np.log10(signal_power(sigLO)/signal_power(sigCh))))

# plot spectrum fter adding LO
plt.psd(
    sigRx,
    Fs=Fs,
    Fc=paramTx.Fc,
    NFFT=4 * 1024,
    sides="twosided",
    label="optical spectrum - Rx + LO",
)
plt.legend(loc="lower left")
plt.title("optical WDM spectrum");
# -

# ### Phase-retrieval stage

# +
# simulate ideal direct-detection optical receiver
Amp = np.abs(sigRx)

# Kramers-Kronig phase-retrieval
phiTime = KramersKronigRx(Amp, Fs)

# optical field reconstruction
sigRx = Amp*np.exp(1j*phiTime)

# remove DC level
sigRx -= np.sqrt(Plo)  # np.mean(sigRx)

# downshift to baseband
sigRx *= np.exp(-1j * (2 * π * FO * t))

# plot spectrum of  the reconstructed field
plt.psd(
    sigRx,
    Fs=Fs,
    Fc=paramTx.Fc,
    NFFT=4 * 1024,
    sides="twosided",
    label="optical spectrum - Rx after KK",
)
plt.legend(loc="lower left")
plt.title("optical WDM spectrum");
# -

# #### Standard receiver processing

# +
# Matched filtering and CD compensation

# Matched filtering
if paramTx.pulse == "nrz":
    pulse = pulseShape("nrz", paramTx.SpS)
elif paramTx.pulse == "rrc":
    pulse = pulseShape(
        "rrc", paramTx.SpS, N=paramTx.Ntaps, alpha=paramTx.alphaRRC, Ts=1 / paramTx.Rs
    )

pulse = pulse / np.max(np.abs(pulse))
sigRx = firFilter(pulse, sigRx)

# plot constellations after matched filtering
pconst(sigRx[0::paramTx.SpS], lim=True, R=3)

# CD compensation
sigRx = edc(sigRx, Ltotal, D, Fc, Fs)

#plot constellations after CD compensation
pconst(sigRx[0::paramTx.SpS], lim=True, R=2)

# Downsampling to 2 sps and re-synchronization with transmitted sequences
sigRx = sigRx.reshape(-1, 1)

# decimation
paramDec = parameters()
paramDec.SpS_in = paramTx.SpS
paramDec.SpS_out = 2
sigRx = decimate(sigRx, paramDec)

symbRx = symbolSync(sigRx, symbTx, 2)
# -

# Power normalization
x = sigRx
d = symbRx

x = x.reshape(len(x), 1) / np.sqrt(signal_power(x))
d = d.reshape(len(d), 1) / np.sqrt(signal_power(d))

# +
# Adaptive equalization
mod = QAMModem(m=paramTx.M)

paramEq = parameters()
paramEq.nTaps = 15
paramEq.SpS = 2
paramEq.mu = [5e-3, 2e-3]
paramEq.numIter = 5
paramEq.storeCoeff = False
paramEq.alg = ["da-rde", "rde"]
paramEq.M = paramTx.M
paramEq.L = [20000, 80000]

y_EQ, H, errSq, Hiter = mimoAdaptEqualizer(x, dx=d, paramEq=paramEq)

discard = int(paramEq.L[0]/2)

#plot constellations after adaptive equalization
pconst(y_EQ[discard:-discard,:], lim=True)

# +
# Carrier phase recovery
paramCPR = parameters()
paramCPR.alg = "bps"
paramCPR.M = paramTx.M
paramCPR.N = 35
paramCPR.B = 64
paramCPR.pilotInd = np.arange(0, len(y_EQ), 20)

y_CPR, θ = cpr(y_EQ, symbTx=d, paramCPR=paramCPR)

y_CPR = y_CPR / np.sqrt(signal_power(y_CPR))

plt.figure()
plt.title("CPR estimated phase")
plt.plot(θ, "-")
plt.xlim(0, len(θ))
plt.grid()

discard = 5000

# plot constellations after CPR
pconst(y_CPR[discard:-discard, :], lim=True)
# -

# #### Evaluate transmission metrics

# +
# correct for (possible) phase ambiguity
for k in range(y_CPR.shape[1]):
    rot = np.mean(d[:, k] / y_CPR[:, k])
    y_CPR[:, k] = rot * y_CPR[:, k]

y_CPR = y_CPR / np.sqrt(signal_power(y_CPR))

ind = np.arange(discard, d.shape[0] - discard)
BER, SER, SNR = fastBERcalc(y_CPR[ind, :], d[ind, :], mod)
GMI, _ = monteCarloGMI(y_CPR[ind, :], d[ind, :], mod)
MI = monteCarloMI(y_CPR[ind, :], d[ind, :], mod)

print("Results:\n")
print("SER: %.2e" % (SER[0]))
print("BER: %.2e" % (BER[0]))
print("SNR: %.2f dB" % (SNR[0]))
print("MI: %.2f bits" % (MI[0]))
print("GMI: %.2f bits" % (GMI[0]));

# +
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def abs_and_phases(sfm):
    """ Divisão do sinal em fase mínima em componentes de amplitudes e fases.
    Args:
        sfm (np.ndarray): Sinal QAM reamostrado com frequência de amostragem maior 
        que a taxa de símbolos.
    Returns:
        data (dict): Dicionário com os arrays contendo as informações de
        amplitudes e fases do sinal.
    """
    amplitudes = np.abs(sfm)
    phases = np.angle(sfm)
    data = {'amplitudes': amplitudes, 'phases': phases}
    return data


def createDataset(sfm, ordem: int,size):
    """O dataset criado pela função é o resultado de uma convolução simples ao longo
    do vetor das amplitudes, para a criação das features, e a fase correspondente à
    amostra de amplitude central da janela em cada passo da convolução.
    Args:
        sfm (np.ndarray): Símbolos do sinal QAM a ser observado.
        ordem (int): Número de amostras de amplitudes a serem observadas para a análise
            de uma amostra de fase.
    Returns:
        data (dict[np.ndarray, np.ndarray]): Dicionário com os arrays contendo as informações de
            amplitudes e fases do sinal.
        X (np.ndarray): Matriz contendo as informações de amplitudes do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
        y (np.ndarray): Vetor coluna contendo as informações de fases do sinal dispostas de tal 
            forma que qualquer algorítmo de regressão de ML pode utilizar como features.
    """
    data = abs_and_phases(sfm)
    amplitudes = data['amplitudes'].copy()
    phases = data['phases'].copy()
    X = np.zeros((size, ordem))
    for n in range(size):
        aux = amplitudes[n:ordem+n]
        X[n] = aux
    y = phases[int(ordem/2):size+int(ordem/2)]
    data['amplitudes'] = data['amplitudes'][int(ordem/2):size+int(ordem/2)]
    data['phases'] = data['phases'][int(ordem/2):size+int(ordem/2)]

    return data, X, y

def train_test_datasets(X,y,size):
    X_train = X[:int(0.8*size)]
    X_test = X[int(0.8*size):]
    y_train = y[:int(0.8*size)]
    y_test = y[int(0.8*size):]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train,X_test, y_test

def ANN_model(X_train, y_train,X_test, y_test,patience = 5):
    stop = EarlyStopping(monitor='val_loss', patience=patience)
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(X_test.shape[1],)))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=300, callbacks=[stop],
          validation_data=(X_test, y_test), batch_size=32)
    return model

def predict_signal(model,dataset,X_test,size):
    preds = model.predict(X_test)
    predicted = dataset['amplitudes'][int(0.8*size):]*np.exp(1j*preds.reshape(-1,))
    return predicted.reshape((1,-1))
# -


