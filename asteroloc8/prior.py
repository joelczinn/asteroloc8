import numpy as np                                            

from scipy.stats import norm, multivariate_normal

from .asteroestimate import detections
from .asteroestimate.detections import locate as loc     

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


class NuPrior(object):
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
        
        # from Pinsonneault et al. 2018
        self.teff_sun = 5772. 
        self.dnu_sun = 135.146                                                                               
        self.numax_sun = 3076.                                                                               
        self.logg_sun = 2.7413e4   
        
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
        numaxsamp = loc.numax_from_JHK(Jsamp, Hsamp, Ksamp, parallaxsamp, mass=mass, AK=AK)                
        numax_median = np.nanmedian(numaxsamp)                                                                     
        numax_std = np.nanstd(numaxsamp)                                                                     
        return (numax_median, numax_std), numaxsamp   
    

    #@staticmethod
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
        
        numax = 10.**(logg)/(self.logg_sun)*self.numax_sun*(teff/self.teff_sun)**(-0.5) 
        return numax
    
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
    