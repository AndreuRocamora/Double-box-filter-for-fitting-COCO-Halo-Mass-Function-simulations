# Double-box-filter-for-fitting-COCO-Halo-Mass-Function-simulations
#COCO simulations from Bose et.al (2016) are fited using the new so called smooth Double Box filter (defined using two k-sharp filters: one weighted at 3/4 and one
weighted at 1/4 with an scale variation).  The name 'smooth', although the filter is not smoothed is used here to distinguish it 
from the simple Double Box filter which is formed using two equally weighted boxes (k-sharp filters) with different scale variations. 

#All the code can be executed without any extra library than numpy, matplotlib and scipy. It will return several figures:

#Figure 1: power spectrum for CDM and for the WDM with a transfer function with the parameters selected above

#Figure 2: variance as a function of mass for k-sharp filter and for CDM (blue) and WDM (orange) 

#Figure 3: derivative of the variance as a function of k for CDM (blue) and WDM (orange)

#Figure 4: simmulation points from Bose et.al (2016)

#Figure 5: fit for the CDM and WDM halo mass function for the simmulation points from Bose et.al (2016)

#Figure 6: variance as a function of mass for smooth-k, smooth Double Box, Double Box and four boxes filter for WDM in addition to a comparison with the approximated integration for the smooth-k filter using the limit trends

#Figure 7: derivative of the variance as a function of k for smooth-k, smooth Double Box, Double Box and four boxes filter for WDM 

#Figure 8: fit for the WDM halo mass function with an smooth-k filter using the simmulation points from Bose et.al (2016)

#Figure 9: fit for the WDM halo mass function with a Double Box filter using the simmulation points from Bose et.al (2016)

#Figure 10: fit for the WDM halo mass function with a four boxes filter using the simmulation points from Bose et.al (2016)

#Figure 11: fit for the WDM halo mass function with the smooth Double Box filter using the simmulation points from Bose et.al (2016)

#Figure 12: one plot with the fited functions for the several filters studied

#Figure 13: one plot with the fited functions for the smooth-k and the smooth Double Box with the deviation from the interpolated simulation points.

#Figure 14: one plot with the fited functions for the smooth-k, smooth Double Box and k-sharp filters with the deviation from the interpolated simulation points.

#Figure 15: one plot with several filtering functions as a function of k*R*c

#Figure 16: spectral indexs for the CDM and WDM power spectrum

#Figure 17: derivative of  the WDM power spectrum to achieve the dependance of k_M with n
