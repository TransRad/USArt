import numpy as np
from MRDLin import MRDOI
import random
import scipy.ndimage 
from numpy.fft import  ifft2, ifftshift


def add_motion_artifacts(ks2d=None, events=200, ratio=0.25, phase_direction=1, lines=None, motion_type=None):
    """
    tuning
    :param ks2d: 2D kspace
    :param events: how many events happened (find random the lines/column)
    :param ratio: ratio of kspace that shift [0 1]
    :param phase_direction: direction (1: horizontal else: vertical)
    :param lines: specific lines/column that need to shift (if need it)
    :param max_events: maximum number of the events
    :param max_shift: maximum shift
    """
    ks2d_copy = ks2d.copy()
    
    ks_new = ks2d_copy
    if lines is None:
        lines = np.zeros(events, dtype=int)
        for i in range(0, events):
            lines[i] = int(random.uniform(0, int(np.shape(ks2d_copy)[0])))
    
    for ii in lines:  # int(step)
        motion_ratio = int(random.uniform(-20, 20))
        ks_motion = motion_im(ks2d, motion_ratio, motion_type)
        if phase_direction == 1:
            ks_new[:, ii] = ks2d_copy[:, ii] * (1-ratio) + ks_motion[:, ii] * ratio 
        else:
            ks_new[ii, :] = ks2d_copy[ii, :] * (1-ratio) + ks_motion[ii, :] * ratio 

    # Reconstructed image
    noisy_im = MRDOI.recon_corrected_kspace(ks_new)

    motion_im2d = noisy_im
    motion_ks2d = ks_new

    return motion_im2d, motion_ks2d, lines


def motion_im(ks2d, ratio, motion_type=None):
    if motion_type is None:
        motion_type = int(random.uniform(0, 2))
    ks_motion = np.zeros(np.shape(ks2d), dtype=complex)   
    if motion_type == 1:
        pivot = np.around(np.shape(ks2d))
        pivot = pivot/2
        pivot = pivot.astype(int)
        ks_motion = rotateImage(ks2d, angle=ratio, pivot=pivot)
    else:
        axis = int(random.uniform(0, 2))
        ks_motion = shift_img_along_axis(ks2d, axis=axis, shift=ratio)
    
    return ks_motion


def rotateImage(ks2d, angle=10, pivot=[0,0]):
    img = ifftshift(ifft2(ifftshift(ks2d)))    
    im_motion = np.zeros(np.shape(img), dtype=complex)   
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = scipy.ndimage.rotate(imgP, angle, reshape=False, order= 0)
    im_motion = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
    kspace = ifftshift(ifft2(ifftshift(im_motion)))
    return kspace


def shift_img_along_axis(ks2d, axis=0, shift=10, constant_values=0):
    """ shift array along a specific axis. New value is taken as weighted by the two distances to the assocaited original pixels.
    CHECKED : Works for floating shift ! ok.
    NOTE: at the border of image, when not enough original pixel is accessible, data will be meaned with regard to additional constant_values. 
    constant_values: value to set to pixels with no association in original image img 
    RETURNS : shifted image. 
    A.Mau. """
    
    img = ifftshift(ifft2(ifftshift(ks2d)))
    intshift = int(shift)
    remain0 = abs( shift - int(shift) )
    remain1 = 1-remain0 #if shift is uint : remain1=1 and remain0 =0
    npad = int( np.ceil( abs( shift ) ) )  #ceil relative to 0. ( 0.5=> 1 and -0.5=> -1 )
    pad_arg = [(0,0)]*img.ndim
    pad_arg[axis] = (npad,npad)
    bigger_image = np.pad( img, pad_arg, 'constant', constant_values=constant_values) 
    
    part1 = remain1*bigger_image.take(np.arange(npad+intshift, npad+intshift+img.shape[axis]) ,axis)
    if remain0==0:
        shifted = part1
    else:
        if shift>0:
            part0 = remain0*bigger_image.take(np.arange(npad+intshift+1, npad+intshift+1+img.shape[axis]) ,axis) 
        else:
            part0 = remain0*bigger_image.take(np.arange(npad+intshift-1, npad+intshift-1+img.shape[axis]) ,axis) 

        shifted = part0 + part1
        
    kspace = ifftshift(ifft2(ifftshift(shifted)))
    return kspace


def random_motion_level():                                                                                               
    """
    Extract random motion level
    :param events: how many events happened (find random the lines/column)
    :param ratio: ratio of kspace that shift [0 1]
    :param phase_direction: direction (1: horizontal else: vertical)
    :param lines: specific lines/column that need to shift (if need it)
    :param max_events: maximum number of the events
    :param max_shift: maximum shift
    """
    min_events = 20
    max_events = 100
    events = int(random.uniform(min_events, max_events))
    min_average = 1
    max_average = 3
    ratio = 1/int(random.uniform(min_average, max_average))
    phase_direction = random.randint(0, 1)
    return events, ratio, phase_direction


def add_guassian_noise_artifacts(ks2d=None, mean=0, sigma_level=0.5):
# Add noise artifacts in the kspace
#
# Output:
# motion_im2d - 2D kspace with noise
# motion_ks2d - 2D reconstructed image with noise
#
# Input:
# ks2d - 2D kspace
# sigma - level of noise suggestion [1 1.5]
    sigma_level = sigma_level*1.5
    ks2d_copy = ks2d.copy()
    noisy_im2d = np.zeros(np.shape(ks2d_copy))
    noisy_ks2d = np.zeros(np.shape(ks2d_copy), dtype=complex)

    sz = np.shape(ks2d_copy)
    point = np.abs(ks2d_copy.mean())*100
    sigma = point*sigma_level
    gauss = np.random.normal(mean, sigma, sz)
    gauss = gauss.reshape(sz)
    noise = np.ones(np.shape(ks2d_copy), dtype=complex) * gauss
    ks_new = ks2d_copy + noise

    # Noisy reconstructed image
    noisy_im = MRDOI.recon_corrected_kspace(ks_new)

    noisy_im2d = noisy_im
    noisy_ks2d = ks_new

    return noisy_im2d, noisy_ks2d


def random_guassian_noise_level():
# Extract random guassian noise level 
#
# Output:
# mean - 0
# sigma_level - [0.2 0.3]
    mean = 0 # keep it 0
    min_rand = 0.2
    max_rand = 0.3
    sigma_level = random.uniform(min_rand, max_rand)
    return mean, sigma_level

