# Import the required modules
# from scipy import complex_
import numpy as np


def recon_corrected_kspace(corrected_kspace=None):
# reconstruction of the kspace 2D
#
# Output:
# im - reconstructed image
#
# Input:
# ks - kspace
    im = np.fft.ifft(corrected_kspace, axis=1)
    im = np.fft.ifft(im, axis=0)
    im = np.fft.ifftshift(im)
    # im = np.abs(im)
    return im



if __name__ == '__main__':
    pass
