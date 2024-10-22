# USArt

This repository contains USArt code for the paper: 
> G. Kanli, D. Perlo, S. Boudissa, Radovan Jiˇr´ı  and O. Keunen.
> "Simultaneous Image Quality Improvement and Artefacts Correction in Accelerated MRI", 2024 MICCAI Workshop on Deep Generative Models, Marrakech, Morocco, 2024 (will be updated upon ).



This paper was accepted to the 21st MICCAI Workshop on Deep Generative Models . You can find the poster presented to the [MICCAI Workshop]() conference under [*Poster_MICCAI-WS_.pdf*](https://github.com/TransRad/MRArt/blob/main/Poster_ISBI24_542.pdf).

## Description
We propose a library named **MRArt**, for MRI Artefacts, that simulates realistic primary artefacts by manipulating the k-space signal, using standard image processing techniques.
MRArt focuses on three degradations commonly encountered in anatomical images:
- Gaussian noise
- Blurriness effect
- Motion artefact

Each of the above degradation are generated with varying levels from low to severe.
The image below illustrates the image domain and the corresponding k-space domain of each degradation types.

#![Alt text](image/img_kspace_ind10_level3.png)

## Getting started
### Requirements

In order to use the library you need to install the following packages:
- pip install matplotlib
- pip install numpy
- pip install pydicom
- pip install scipy
- pip install skimage


You can find instructions on loading the data and utilizing the MRArt library in the Jupyter notebook titled **Examples.ipynb**.

## Support
If you have any questions, feel free to reach out to our support team at imaging@lih.lu.

## Roadmap
We plan to introduce a 3D version of the library in the future, featuring additional degradation types.

## Authors
Selma BOUDISSA, Georgia KANLI, Daniele PERLO, Thomas JAQUET and Olivier KEUNEN.


## Citation 
If you find **MRArt** useful in your research, please use the following for citation.

S. Boudissa, G. Kanli, D. Perlo, T. Jaquet and O. Keunen, "Addressing Artefacts in Anatomical MR Images: A k-space-Based Approach," 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, pp. 1-5, doi: 10.1109/ISBI56570.2024.10635199. 



