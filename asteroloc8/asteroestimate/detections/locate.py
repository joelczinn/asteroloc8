"""
locate.py

Description
-----------
Locates numax from JHK photometry or luminosity and teff.

Contributor(s)
--------------
Ted Mackereth   - author of code ported from https://github.com/jmackereth/asteroestimate.git
Alex Lyttle     - removed unnecessary functions and renamed file

"""
import numpy as np
from scipy.stats import chi2, multivariate_normal, norm
from scipy.interpolate import interp1d
import asteroloc8.asteroestimate.bolometric.polynomial as polybcs
import asteroloc8.asteroestimate.parsec.grid as grid
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

numax_sun = 3150 # uHz
dnu_sun = 135.1 # uHz
teff_sun = 5777 # K
taugran_sun = 210 # s
teffred_sun = 8907 # K

obs_available = ['kepler-sc', 'kepler-lc', 'tess-ffi']

def numax(mass,teff,rad):
    """
    nu_max from scaling relations
    INPUT:
        mass - stellar mass in Msun
        teff - Teff in K
        rad - stellar radius in Rsun
    OUTPUT:
        numax - numax from scaling relations in uHz
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    return numax_sun*mass*(teff/teff_sun)**-0.5*rad**-2.

def dnu(mass,rad):
    """
    delta nu from scaling relations
    INPUT:
        mass - stellar mass in Msun
        rad - stellar radius in Rsun
    OUTPUT:
        deltanu - delta nu based on scaling relations in uHz
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    return dnu_sun*(mass*rad**-3.)**0.5

def luminosity(rad,teff):
    """
    luminosity in L_sun from scaling relations
    INPUT:
        rad - stellar radius in Rsun
        teff - Teff in K
    OUTPUT:
        lum - luminosity in L_sun from scaling relations
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    return rad**2*(teff/teff_sun)**4.

def teffred(lum):
    """
    T_red, temperature on red edge of Delta Scuti instability strip (see Chaplin+ 2011)
    """
    return teffred_sun*(lum)**-0.093

def Kmag_to_lum(Kmag, JK, parallax, AK=None, Mbol_sun=4.67):
    """
    convert apparent K mag, J-K colour and parallax into luminosity
    INPUT:
        Kmag - apparent K band magnitude
        JK - J-K colour
        parallax - parallax in mas
        AK - extinction in K band
        Mbol_sun - the solar bolometric magnitude
    OUTPUT:
        luminosity in L_sun
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    BCK = polybcs.BCK_from_JK(JK)
    if AK is None:
        MK = Kmag-(5*np.log10(1000/parallax)-5)
    else:
        MK = Kmag -(5*np.log10(1000/parallax)-5) - AK
    Mbol = BCK+MK
    lum = 10**(0.4*(Mbol_sun-Mbol))
    return lum

def J_K_Teff(JK, FeH=None, err=None):
    """
    Teff from J-K colour based on Gonzalez Hernandez and Bonifacio (2009)
    INPUT:
        JK - J-K colour
        FeH - the [Fe/H] for each entry
        err - error on JK (optional)
    OUTPUT:
        T_eff - the effective temperature
        T_eff_err - error on T_eff
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    if FeH is None:
        #include a prior on feh? for now just assume solar
        theff = 0.6524 + 0.5813*JK + 0.1225*JK**2.
        if err is not None:
            b2ck=(0.5813+2*0.1225*JK)
            a = (5040*b2ck/(0.6524+JK*b2ck)**2)**2
            tefferr = np.sqrt(a*err**2)
    else:
        theff = 0.6524 + 0.5813*JK + 0.1225*JK**2. - 0.0646*JK*FeH + 0.0370*FeH + 0.0016*FeH**2.
    if err is not None:
        return 5040/theff, tefferr
    return 5040/theff

def numax_from_JHK(J, H, K, parallax, mass=1., return_samples=False, AK=None):
    """
    predict frequency at maximum power from 2MASS photometry and Gaia parallax
    INPUT:
        J, H, K - 2MASS photometry
        parallax - parallax from Gaia/other in mas
        mass - an estimate of the stellar mass, can either be a constant (float) for the whole sample, samples for each star based on some prior (N,N_samples), or use 'giants'/'dwarfs' for a prior for these populations
        return_samples - return the samples of numax based on the input mass samples
        return_lum - return the luminosity based on JHK photometry
        AK - the K band extinction
    OUTPUT:
        numax - the predicted numax in uHz
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    tlum = Kmag_to_lum(K, J-K, parallax, AK=AK, Mbol_sun=4.67) #luminosity in Lsun
    if AK is not None:
        tteff = J_K_Teff(J-K-1.5*AK) #teff in K
    else:
        tteff = J_K_Teff(J-K)
    tteff /= teff_sun
    trad = np.sqrt(tlum/tteff**4)
    if isinstance(mass, (int, float,np.float32,np.float64,np.ndarray)):
        tmass = mass
        tnumax = numax(tmass, tteff*teff_sun, trad)
        return tnumax
    elif mass == 'giants':
        ndata = len(J)
        msamples = np.random.lognormal(mean=np.log(1.2), sigma=0.4, size=ndata*100)#sample_kroupa(ndata*100)
        tnumax = numax(msamples, np.repeat(tteff,100)*teff_sun, np.repeat(trad,100))
        tnumax =  tnumax.reshape(ndata,100)
        if return_samples:
            return tnumax
        return np.median(tnumax, axis=1)


def numax_from_luminosity_teff(luminosity, teff, mass=1., return_samples=False, AK=None):
    """
    predict frequency at maximum power from 2MASS photometry and Gaia parallax
    INPUT:
        luminosity - luminosity in L_sun
        teff - effective temperature in K
        mass - an estimate of the stellar mass, can either be a constant (float) for the whole sample, samples for each star based on some prior (N,N_samples), or use 'giants'/'dwarfs' for a prior for these populations
        return_samples - return the samples of numax based on the input mass samples
        return_lum - return the luminosity based on JHK photometry
        AK - the K band extinction
    OUTPUT:
        numax - the predicted numax in uHz
    HISTORY:
        27/04/2020 - written - J T Mackereth (UoB)
    """
    tlum = luminosity #teff in K
    tteff = teff/teff_sun
    trad = np.sqrt(tlum/tteff**4)
    if isinstance(mass, (int, float,np.float32,np.float64)):
        tmass = mass
        tnumax = numax(tmass, tteff*teff_sun, trad)
        return tnumax
    elif mass == 'giants':
        # ndata = len(J)  # <--- changed to luminosity below
        ndata = len(luminosity)
        msamples = np.random.lognormal(mean=np.log(1.2), sigma=0.4, size=ndata*100)#sample_kroupa(ndata*100)
        tnumax = numax(msamples, np.repeat(tteff,100)*teff_sun, np.repeat(trad,100))
        tnumax =  tnumax.reshape(ndata,100)
        if return_samples:
            return tnumax
        return np.median(tnumax, axis=1)
