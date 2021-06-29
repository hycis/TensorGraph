'''
#------------------------------------------------------------------------------
PyRadiomics functions
Routines to generate radiomics LoG and Wavelet transforms from inputs
Louis Lee 11-10-2018
#-------------------------------------------------------------------------------
API details:
    Version: ML_BRAIN_TUMOR_v2.0.0
    Internal identifier: Model5
Script details:
    Version: v1.0.0
    Internal identifier: PyRadiomics.py
#-------------------------------------------------------------------------------
'''
# Python2 compatibility
from __future__ import print_function

import numpy as np
import SimpleITK as sitk
import radiomics.imageoperations

# Fixed list of LoG and wavelets to generate
#sigma_list = [1.0, 2.0, 3.0, 4.0, 5.0]
#n_wavelets = 8
sigma_list = [1.0, 3.0, 5.0]
n_wavelets = 4

def getNumFeatures(imgsize_in, nchannels=1):
    '''
    Accessory function to return # feature maps
    Input: 3D image size + channels tuple (z,y,x,# channels)
    Output: 3D image size + channels tuple (z,y,x,# radiomics channels)
    '''
    return imgsize_in[:-1] + (nchannels*(len(sigma_list) + n_wavelets),)

def radiomicsLoGWavelet(np_img):
    '''
    Function to calculate & return radiomics transformations
    Input: FP NumPy image of dimension (z,y,x)
    Output: Tuple of (LoG, Wavelet) FP NumPy image of dimension
            (z,y,x, # radiomics channels). E.g. LoG has dim (z,y,x,#LoG channels)
    '''
    sitk_img = sitk.GetImageFromArray(np_img.astype(np.float32))
    sitk_msk = sitk.GetImageFromArray(np.ones(np_img.shape, np.float32))

    LoG = radiomics.imageoperations.getLoGImage( \
        sitk_img ,sitk_msk, sigma=sigma_list)
    wavelet = radiomics.imageoperations.getWaveletImage(sitk_img, sitk_msk, \
        force2D=True, force2Ddimension=0)

    LoG_np = []
    for isample in range(len(sigma_list)):
        LoG_sitk,_,_ = next(LoG)
        LoG_img = sitk.GetArrayFromImage(LoG_sitk)
        LoG_np.append(np.expand_dims(LoG_img, axis=-1))
    LoG_np = np.concatenate(LoG_np, axis=-1)

    wavelet_np = []
    i = 0
    for isample in range(n_wavelets):
        wavelet_sitk,_,_ = next(wavelet)
        wavelet_img = sitk.GetArrayFromImage(wavelet_sitk)
        wavelet_np.append(np.expand_dims(wavelet_img, axis=-1))
    wavelet_np = np.concatenate(wavelet_np, axis=-1)

    return LoG_np, wavelet_np

def getFeatures(batch_imgchannels):
    '''
    Function to transform batch of input images into batch of radiomics transforms
    Input: NumPy images of dimension (batchsize,z,y,x,# channels)
    Output: NumPy images of dimension (batchsize,z,y,x,# output channels) where
            # output channels = # radiomics channels per input channel x # input channels
    '''
    batchsize = batch_imgchannels.shape[0]
    nchannels = batch_imgchannels.shape[-1]

    # Iterate over each sample in batch
    output = []
    for ibatch in range(batchsize):
        imgs = batch_imgchannels[ibatch,:,:,:,:]
        # Iterate over each channel in sample
        features = []
        for ichannel in range(nchannels):
            LoG, wavelet = radiomicsLoGWavelet(imgs[:,:,:,ichannel])
            features.append(np.concatenate([LoG, wavelet], axis=-1))
        # Stack all radiomics output from each channel to last dim
        output.append(np.concatenate(features, axis=-1))
    # Stack all batch outputs to 1st dim
    output = np.stack(output, axis=0)

    return output
