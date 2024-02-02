# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:35:36 2022

@author: Kseniia Volkova
"""

import numpy as np
import scipy.optimize as opt

# CONSTANTS
# Spin density in an immersion oil
rho_oil = 50    # 1/nm^3

# Reduced gyromagnetic ratios
# Gyromagnetic ratio NV, electron
gamma_NV = 2.8024 * 1e6    # Hz/G
gamma_e = 2 * np.pi * gamma_NV / 1e2 # rad/s*T
# Gyromagnetic ratio 1H, proton
gamma_1H = 26.7522128 * 1e7 * 1e-4 / (2 * np.pi)    # rad/s*T -> Hz/G
gamma_13C = 6.728284 * 1e7 * 1e-4 / (2 * np.pi)    # rad/s*T -> Hz/G
gamma_125Te = 8.5108404 * 1e7 * 1e-4 / (2 * np.pi)    # rad/s*T -> Hz/G
gamma_123Te = 7.059098 * 1e7 * 1e-4 / (2 * np.pi)    # rad/s*T -> Hz/G
gamma_19F = 25.18148 * 1e7 * 1e-4 / (2 * np.pi)    # rad/s*T -> Hz/G
gamma_6Li = 3.9371709 * 1e7 * 1e-4 / (2 * np.pi)    # rad/s*T -> Hz/G
gamma_7Li = 10.3977013 * 1e7 * 1e-4 / (2 * np.pi)    # rad/s*T -> Hz/G
gamma_31P = 10.8394 * 1e7 * 1e-4 / (2 * np.pi)    # rad/s*T -> Hz/G



# FUNCTIONS FOR THE DATA IMPORT

def qudi_reader(filename, columns):
    """Imports qudi data with or without alternating columns.
    
    Input:
    filename: Qudi data file.
    columns: 2 - for the data without alternating signal; 3 - for the data with alternating signal.
    """
    reader = np.loadtxt(filename)
    time = reader[:,0] * 1e6    # us
    signal = reader[:,1]    # Signal
    if columns == 3:
        signal_alt = reader[:,2]    # Alternating signal
        level = ((signal + signal_alt)/2).mean()
        delta = (signal + (np.max(signal_alt) - signal_alt + level)) / 2
        if delta.mean() < 0:
            delta = -delta
        return time, signal, signal_alt, delta
    return time, signal

def mp_reader(filename):
    """Imports .mp data from the counter.
    
    Input:
    filename: Counter data file.
    """
    with open(filename, 'r') as inputfile:
        lines = []
        for line in inputfile:
            lines.append(line)
        yarray = np.array(lines[54:])
        counts = np.array([int(numeric_string) for numeric_string in yarray])
        time = []
        for i in range(len(counts)):
            time.append(i * 100) # 1 bin = 100 ps
    return time, counts

def csvReader(filename):
    """Imports .csv data from the spectrometer.
    
    Input:
    filename: Spectrometer data file.
    """
    with open(filename, "r") as inputfile:
        reader = np.loadtxt(inputfile, delimiter=',') # reading *.csv file with defined delimiter between values
        xdata = reader[:,5] # np.array of float values of the wavelength
        ydata = reader[:,4] # np.array of float values of the signal
    return xdata,ydata



# FUNCTIONS FOR THE FITTING
def saturation_func(x, a, b):
    """Defines the linear function for the fitting.
    
    Input:
    x: Laser power.
    a: Slope.
    b: Shift.
    """
    return a * x + b

def T2_func(x, a, b, c):
    """Defines the exponential function for the fitting.
    
    Input:
    x: Time.
    a: Highest point of the exponential decay.
    b: Lifetime/T2.
    c: Level of the exponential tail.
    """
    
    return a * np.exp(-(x * (1 / b))) + c

def double_exp(x, a, b, a1, b1, c):
    """Defines the double exponential function for the fitting.
    
    Input:
    x: Time.
    a: Highest point of the exponential decay.
    b: Lifetime/T2.
    a1: Second amplitude.
    b1: Second lifetime/T2.
    c: Level of the exponential tail.
    """
    
    return a * np.exp(-(x * (1 / b))) + a1 * np.exp(-(x * (1 / b1))) + c

def cos_func(x, a, b, c, d):
    """Defines the Rabi function for the fitting.
    
    Input:
    x: Time.
    a: Amplitude.
    b: Frequency.
    c: Phase shift.
    d: Baseline.
    """
    
    return a * np.cos(2 * np.pi * b * x + c) + d

def Lorentzian_func(x, a, b, c, d):
    """Defines the Rabi function for the fitting.
    
    Input:
    x: Time.
    a: Amplitude.
    b: Center.
    c: Sigma.
    d: Baseline.
    """
    return (a * c**2) / ((x - b)**2 + c**2) + d

def double_Lorentzian_func(x, a, b, c, a1, b1, c1, d):
    """Defines the Rabi function for the fitting.
    
    Input:
    x: Time.
    a: Amplitude.
    b: Center.
    c: Sigma.
    d: Baseline.
    a1: Second amplitude.
    b1: Second center.
    c1: Second sigma.
    """
    return ((a * c**2) / ((x - b)**2 + c**2)) + ((a1 * c1**2) / ((x - b1)**2 + c1**2)) + d

def Lorentzian_decay_func(x, a, b, c, a1, d):
    """Defines the Rabi function for the fitting.
    
    Input:
    x: Time.
    a: Amplitude.
    b: Center.
    c: Sigma.
    a1: Highest point of the decay.
    d: Baseline.
    """
    return (a * c**2) / ((x - b)**2 + c**2) + a1 * x + d

def double_Lorentzian_decay_func(x, a, b, c, a1, b1, c1, a2, d):
    """Defines the Rabi function for the fitting.
    
    Input:
    x: Time.
    a: Amplitude.
    b: Center.
    c: Sigma.
    a1: Second amplitude.
    b1: Second center.
    c1: Second sigma.
    a2: Highest point of the decay.
    d: Baseline.
    """
    return ((a * c**2) / ((x - b)**2 + c**2)) + ((a1 * c1**2) / ((x - b1)**2 + c1**2)) + a2 * x + d
    
def decay_func(x, a, b):
    """Defines linear decay for the fitting.
    
    Input:
    x: Time.
    a:.
    b:.
    """
    return a * x + b



# FUNCTIONS FOR THE ANALYSIS

def thetaB(ODMR_delta):
    i1 = ODMR_delta / 2870 * 1e6 # Hz
    i2 = np.acos(i1)
    return (i2/2)*(180/np.pi)

def norm(XY8_signal, Rabi_signal, XY8_level):
    """Defines the XY8 signal normalized to the contrast of the Rabi oscillation.
    
    Input:
    XY8_signal: XY8 fluorescence data.
    Rabi_signal: Rabi flurescence data.
    XY8_level: level = ((XY8_signal + XY8_alternative_signal) / 2).mean().
    """
    Rabi_amplitude = (np.amax(Rabi_signal) - np.amin(Rabi_signal)) / 2
    XY8_norm = (XY8_signal - XY8_level + Rabi_amplitude) / (2 * Rabi_amplitude)
    return XY8_norm

def deconv(N, XY8_norm, XY8_time):
    """Calculates the power spectral density S in Hz units.
    
    Input:
    N: Number of laser pulses.
    XY8_norm: Normalized XY8 signal to the Rabi contrast.
    XY8_time: Spacing tau between MW pulses.
    """
    XY8_tau = XY8_time / 1e6 # s
    Chi = -np.log(2 * XY8_norm - 1)
    Chi = np.array(Chi)
    Spectral_dens = ((np.pi**2) / 4) * (1 / (N * XY8_tau)) * Chi / 1e3 # kHz
    return Spectral_dens

def fwhm(x, y):
    """Calculates the width of the peak.
    
    Input:
    x: Frequencies.
    y: Function.
    """
    half_max = np.max(y) / 2    
    d = np.sign(half_max - np.array(y[0:-1])) - np.sign(half_max - np.array(y[1:]))
    left_index = np.where(d > 0)[0]
    right_index = np.where(d < 0)[-1]
    FWHM = x[right_index[0]] - x[left_index[0]]
    return FWHM

def depth(B, gamma_n, rho):
    """Calculates the depth of the NV center for given spin density.
    
    Input:
    B: Magnetic field fluctuations, T.
    gamma_n: Reduced gyromagnetic ratio, Hz/T.
    rho: Spin density, 1/nm^3.
    """
    # Vacuum permeability
    mu_0 = 4 * np.pi * 1e-7
    # Planck's constant
    h = 6.62607e-34
    rho *= 1e27
    d = (rho * mu_0 * mu_0 * h * h * gamma_n * gamma_n * 5 / (1536 * np.pi * B * B))**(1 / 3)
    NV_depth = d * 1e9 # nm
    return NV_depth

def Lifetime_exp(xdata, ydata, lifetime):
    """Calculates PL lifetime from a counter's data.
    
    Input:
    xdata: Time, ps.
    ydata: Counts.
    lifetime: Expected lifetime, s.
    """
    # Searches for the highest counts of the measured data
    max_index = int(np.where(ydata == max(ydata))[0])
    # Data fitting
    yfit = np.array(ydata[max_index:])
    # For data fitting you need to move highest point to zero time
    xfit = np.array([i * 1e-10 for i in range(0, len(np.array(xdata[max_index:])))])
    # Initial guess for the parameters [highest point, lifetime, level of the tail]
    p0 = np.array([max(yfit), lifetime, int(np.average(yfit[-100:-5]))])
    # Data fitting with optimal values and the estimated covariance of popt as an output
    popt, pcov = opt.curve_fit(T2_func, xfit, yfit, p0) 
    lt = popt[1] * 1e9 # ns
    # Standard deviation error
    perr = np.sqrt(np.diag(pcov))
#    print('  Lifetime, ns: ', lt, "\u00B1", perr[1] * 1e9)
    return xfit, yfit, lt, popt, perr

def Lifetime_double_exp(xdata, ydata, lifetime):
    """Calculates PL lifetime with double exponential fitting from a counter's data.
    
    Input:
    xdata: Time, ps.
    ydata: Counts.
    lifetime: Expected lifetime, s.
    """
    # Searches for the highest counts of the measured data
    max_index = int(np.where(ydata == max(ydata))[0])
    # Data fitting
    yfit = np.array(ydata[max_index:])
    # For data fitting you need to start from 0 time not the one you have for the highest point from the measurement
    xfit = np.array([i * 1e-10 for i in range(0, len(np.array(xdata[max_index:])))])
    # Initial guess for the parameters [highest point, lifetime, level of the tail]
    p0 = np.array([max(yfit / 2), lifetime, max(yfit) / 2, lifetime, int(np.average(yfit[-100:-5]))])
    # Data fitting with optimal values and the estimated covariance of popt as an output
    popt, pcov = opt.curve_fit(double_exp, xfit, yfit, p0) 
    lt = popt[1] * 1e9 # ns
    lt2 = popt[3] * 1e9 # ns
    # Standard deviation error
    perr = np.sqrt(np.diag(pcov))
#    print('  Lifetime, ns: ', lt, "\u00B1", perr[1] * 1e9)
#    print('  Lifetime2, ns: ', lt2, "\u00B1", perr[3] * 1e9)
    return xfit, yfit, lt, lt2, popt, perr



# FUNCTIONS THAT WILL PRINT OR PLOT
def Estimation(nor, B_ext, gamma, gamma2, gamma_name, gamma2_name):
    """Calculates and prints estimated values for the XY8 measurements.
    
    Input:
    nor: Number of measured resonances.
    B_ext: Applied external magnetic field.
    gamma: Isotope gyromagnetic ratio.
    """
    print('Magnetic field projection on the NV center axis:', round(B_ext,2), 'G\n')
    tau = 1e6 / (2 * gamma * B_ext)    # us
    Larm_freq = gamma * 1e-3 * B_ext    # kHz
    print(gamma_name + ' gyromagnetic ratio:', round((gamma * 1e-3),2), 'kHz/G')
    print(gamma_name + ' Larmor frequency:', round(Larm_freq,2), 'kHz')
    print(gamma_name + ' expected value of tau:', round(tau,2), '\u03BCs')
    if nor == 2: # Isotope with lower gyromagnetic ratio
        tau2 = 1e6 / (2 * gamma2 * B_ext)    # us
        Larm_freq2 = gamma2 * 1e-3 * B_ext    # kHz
        print('\nFor the second signal:')
        print(gamma2_name + ' gyromagnetic ratio:', round((gamma2 * 1e-3),2), 'kHz/G')
        print(gamma2_name + ' Larmor frequency:', round(Larm_freq2,2), 'kHz')
        print(gamma2_name + ' expected value of tau:', round(tau2,2), '\u03BCs')
        return tau, Larm_freq, tau2, Larm_freq2
    else:
        return tau, Larm_freq