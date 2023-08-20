"""
Multiresolution bilateral filter
"""

import cv2
import numpy as np
import pywt
from skimage.restoration import denoise_wavelet

    
def multires_bilateral(img, d=1, sigmaColor=1.8, sigmaSpace=2, wavelet_type='sym2',
                wavelet_levels=2, method='VisuShrink',mode='soft'):
        """ Multiresolution bilateral denoising
        Parameters
        ----------
        img            = RGB color image
        d              = Diameter of each pixel neighborhood
        sigmaColor     = Color fall-off value
        sigmaSpace     = Spatial fall-pff value
        wavelet_type   = Wavelet transform type
        wavelet_levels = Levels of decomposition
        method         = Wavelet thresholding method
        mode           = Additional scikit denoising argument
        
        Returns
        ----------
        img_out        = Denoised image
        """
        
        # Wavelet transform
        # coeffs[0] = Approximation sub-band
        # coeffs[1:] = Detail sub-bands
        coeffs = pywt.wavedec2(img.transpose(2,0,1), wavelet = wavelet_type, 
                               level = wavelet_levels)

        apprx = coeffs[0]
        dcoeffs = coeffs[1:]
        
        # Iterating detail sub-bands.
        for level in dcoeffs:
            
            # Applying bilateral denoising to the approximation sub-band.
            apprx_bilateral = np.zeros_like(apprx)
            for i in range(apprx.shape[0]):
                apprx_bilateral[i] = cv2.bilateralFilter(np.float32(apprx[i]), 
                            d, sigmaColor, sigmaSpace)
                                
            # Applying wavelet-threshold denoising to detail sub-bands.
            denoised_detail = [denoise_wavelet(channel, method=method, 
                                mode=mode, rescale_sigma=True) for channel in level]
            
            # Approximation and detail sub-band are joined.
            coeffs_rec = [apprx_bilateral] + [denoised_detail]
            apprx = pywt.waverec2(coeffs_rec, wavelet_type)
            
        # Final bilateral filtering
        img_out = apprx.transpose(1,2,0)
        img_out = cv2.bilateralFilter(np.float32(img_out), d, 
                                      sigmaColor, sigmaSpace)
        
        # Removing extra rows and columns in a img_out
        if img.shape[0] != img_out.shape[0]:
            img_out = np.delete(img_out, img_out.shape[0]-1, 0)
        if img.shape[1] != img_out.shape[1]:
            img_out = np.delete(img_out, img_out.shape[1]-1, 1)
            

        
        return img_out


    
                