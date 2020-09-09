import numpy as np                                            

from scipy.stats import norm, multivariate_normal

# these are real spec. and phot. data from an anonymous TESS star with measured numax of ~30uHz, with made-up uncertainties.
def get_gaiascalnmx():
    nup = NuPrior(plx=0.44, plx_err=0.01, jmag=10.64, jmag_err=0.01, hmag=10.134, hmag_err=0.01, kmag=10.02, kmag_err=0.01)
    print('(numax_median, numax_std, numax_samples) from gaiascalnmx:')
    print(nup.gaiascalnmx(mass='giants'))
    
def get_specnmx():
    nup = NuPrior(teff_spec=4900., teff_spec_err=100., logg_spec=2.4, logg_spec_err=0.1)
    print('(numax_median, numax_std, numax_samples) from gaiascalnmx:')
    print(nup.specnmx())
    
#get_gaiascalnmx() 
#get_specnmx()


class Asteroseismology:
    """
    A class containing asteroseismic relations and constants.

    TODO: class methods to change constants

    HISTORY:
        09/09/2020 - written - A J Lyttle
    """
    # solar "constants" as class globals
    # from Pinsonneault et al. 2018
    teff_sun = 5772. 
    dnu_sun = 135.146                                                                               
    numax_sun = 3076.                                                                               
    logg_sun = 4.44

    # Ted's constants:
    # numax_sun = 3150 # uHz
    # dnu_sun = 135.1 # uHz
    # teff_sun = 5777 # K

    def logg(self, mass, rad):
        """
        HISTORY:
            09/09/2020 - written - A J Lyttle
        """
        return self.logg_sun + np.log10(mass) - 2 * np.log10(rad)

    def numax(self, logg, teff):
        '''
        Return an expected numax given a log g and teff                                                   
        INPUTS:                                                                                           
        self.logg, self.logg_spec : float, float
         log10 surface gravity and uncertainty [cgs].                                                                     
        self.teff_spec, self.teff_spec_err : float, float                                                                                        
         effective temperature and uncertainty [K].                                                                         
        [ emp : bool ]                                                                                      
        OUTPUTS:                                                                                             
        numax : float                                                                                       
         Frequency of maximum oscillation [muhz].

        '''
        
        numax = 10.**(logg - self.logg_sun) * self.numax_sun * (teff/self.teff_sun)**(-0.5) 
        return numax

class NuPrior(Asteroseismology):
    '''                                                                                              
    Provide guesses for numax using three different methods and also optionally numax prior distributions.
    1) specnmx()
     Uses spectroscopic log g + spectroscopic temperature.
    2) gaiascalnmx()
     Uses Gaia parallax + apparent magnitude + bolometric correction + photometric temperature + optional extinction.
    3) gaiamlnmx():
     Uses a data-driven approach to map Gaia luminosity to numax.
    '''
  
    def __init__(self, plx=None, plx_err=None, logg_spec=None, logg_spec_err=None, teff_spec=None, teff_spec_err=None,
                 jmag=None, jmag_err=None, hmag=None, hmag_err=None, kmag=None, kmag_err=None):
        ''' 
        INPUTS:                                                                                              
        [ plx, plx_err : float, float ]
         Parallax and uncertainty [mas]. Default None.
        [ logg_spec, logg_spec_err : float, float ]
         Spectroscopic log g and uncertainty [cgs]. Default None.
        [ teff_spec, teff_spec_err : float, float ]
         Spectroscopic temperature and uncertainty [K]. Default None.  
        [ jmag, jmag_err : float, float ]
         J-band magnitude and uncertainty [mag]. Default None.
        [ hmag, hmag_err : float, float ]
         H-band magnitude and uncertainty [mag]. Default None. 
        [ kmag, kmag_err : float, float ]
         K-band magnitude and uncertainty [mag]. Default None.
        HISTORY:                                                                                            
        Created 8 sep 20
        Joel Zinn (j.zinn@unsw.edu.au)
        '''
        self.plx = plx
        self.plx_err = plx
        self.logg_spec = logg_spec
        self.logg_spec_err = logg_spec_err
        self.teff_spec = teff_spec
        self.teff_spec_err = teff_spec_err
        
        self.jmag = jmag
        self.jmag_err = jmag_err
        self.hmag = hmag
        self.hmag_err = hmag_err
        self.kmag = kmag
        self.kmag_err = kmag_err

    def BCK_from_JK(self, JK):
        """based on a simple fit to Houdashelt+2000 Table 5 
        HISTORY:
            27/04/2020 - written - J T Mackereth (UoB)
        """
        coeff = np.array([-1.27123055,  3.69172478,  0.11070501])
        poly = np.poly1d(coeff)
        out = poly(JK)
        return out

    def gaiascalnmx(self, mass=1., AK=None, N_samples=1000):                                     
        """                                                                                                 
        Evaluate a prior on numax based on 2MASS magnitudes and Gaia parallax                               
        INPUTS:                                                                                              
        [ plx, plx_err, jmag, jmag_err, hmag, hmag_err, kmag, kmag_err ] : [ float, float, float, float, float, float, float, float ]
         These need to be defined in __init__().
        [ mass : float ]
         Optional mass prior option (not yet implemented!!!). Default 1.               
        [ AK : float ]
         Optional K band extinction. Default None.                                                               
        [ N_samples : int ]
         Number of samples from the prior to take and then return. Default 1000.        
        OUTPUTS:                                                                                             
        (numax_median, numax_std), numax_samp : (float, float), float ndarray
         Numax summary stats. and sample distribution [uHz].
        HISTORY:                                                                                            
        Written - Mackereth - 08/09/2020 (UoB @ online.tess.science)
        Modified JCZ 8 sep 20
        """                                                                                                 
        means = np.array([self.jmag, self.hmag, self.kmag, self.plx])                                                   
        cov = np.zeros((4,4))                                                                               
        cov[0,0] = self.jmag_err**2                                                                                  
        cov[1,1] = self.hmag_err**2                                                                                  
        cov[2,2] = self.kmag_err**2                                                                                  
        cov[3,3] = self.plx_err**2                                                                           
        multi_norm = multivariate_normal(means, cov)                                                        
        samples = multi_norm.rvs(size=N_samples)                                                            
        Jsamp, Hsamp, Ksamp, parallaxsamp = samples[:,0], samples[:,1], samples[:,2], samples[:,3]          
        numaxsamp = self.numax_from_JHK(Jsamp, Hsamp, Ksamp, parallaxsamp, mass=mass, AK=AK)                
        numax_median = np.nanmedian(numaxsamp)                                                                     
        numax_std = np.nanstd(numaxsamp)                                                                     
        return (numax_median, numax_std), numaxsamp   

    def Kmag_to_lum(self, Kmag, JK, parallax, AK=None, Mbol_sun=4.67):
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
        BCK = self.BCK_from_JK(JK)
        if AK is None:
            MK = Kmag-(5*np.log10(1000/parallax)-5)
        else:
            MK = Kmag -(5*np.log10(1000/parallax)-5) - AK
        Mbol = BCK+MK
        lum = 10**(0.4*(Mbol_sun-Mbol))
        return lum

    def J_K_Teff(self, JK, FeH=None, err=None):
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

    def numax_from_JHK(self, J, H, K, parallax, mass=1., return_samples=False, AK=None):
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
        tlum = self.Kmag_to_lum(K, J-K, parallax, AK=AK, Mbol_sun=4.67) #luminosity in Lsun
        if AK is not None:
            tteff = self.J_K_Teff(J-K-1.5*AK) #teff in K
        else:
            tteff = self.J_K_Teff(J-K)
        tteff /= self.teff_sun
        trad = np.sqrt(tlum/tteff**4)
        if isinstance(mass, (int, float, np.float32, np.float64, np.ndarray)):
            tlogg = self.logg(mass, trad)
            tnumax = self.numax(tlogg, tteff*self.teff_sun)
            return tnumax
        elif mass == 'giants':
            ndata = len(J)
            msamples = np.random.lognormal(mean=np.log(1.2), sigma=0.4, size=ndata*100)#sample_kroupa(ndata*100)
            loggsamples = self.logg(msamples, np.repeat(trad,100))
            tnumax = self.numax(loggsamples, np.repeat(tteff,100)*self.teff_sun)
            tnumax =  tnumax.reshape(ndata,100)
            if return_samples:
                return tnumax
            return np.median(tnumax, axis=1)
 
    def specnmx(self, N_samples=1000):                                                         
        '''                                                                                                 
        Return an expected numax, uncertainty, and numax samples, given a log g and teff                                                   
        INPUTS:                                                                                           
        self.logg, self.logg_spec : float, float
         log10 surface gravity and uncertainty [cgs].                                                                     
        self.teff_spec, self.teff_spec_err : float, float                                                                                        
         effective temperature and uncertainty [K].                                                                                    
        [ N_samples : int ]
         Number of samples to draw for numax samples. Default 1000.
        OUTPUTS:                                                                                             
        (numax_median, numax_std), numax_samp : (float, float), float ndarray
         Numax summary stats. and sample distribution [uHz].
         '''  
        #assert (not self.logg_spec)
        #assert (not None self.logg_spec_err)
        #assert is not None self.teff_spec
        #assert is not NOne self.teff_spec_err
        assert self.logg_spec > -99
        assert self.logg_spec_err > 0
        assert self.teff_spec > 0
        assert self.teff_spec_err > 0
        
        means = np.array([self.logg_spec, self.teff_spec])     
        cov = np.zeros((2,2))                                                                               
        cov[0,0] = self.logg_spec_err**2                                                                                  
        cov[1,1] = self.teff_spec_err**2                                                                                                                                                           
        multi_norm = multivariate_normal(means, cov)                                                        
        samples = multi_norm.rvs(size=N_samples)                                                            
        logg_samp, teff_samp = samples[:,0], samples[:,1]          
        numaxsamp = self.numax(logg_samp, teff_samp)          
        numax_median = np.median(numaxsamp)                                                                     
        numax_sigma = np.std(numaxsamp)                                                                     
        return (numax_median, numax_sigma), numaxsamp   
