# USArt

This repository contains USArt code for the paper: 
> G. Kanli, D. Perlo, S. Boudissa, R. Jirik  and O. Keunen.
> "Simultaneous Image Quality Improvement and Artefacts Correction in Accelerated MRI", 2024 MICCAI Workshop on Deep Generative Models, Marrakech, Morocco, 2024 (will be updated once paper published in proceedings ).



This paper was accepted to the **27th MICCAI Workshop on Deep Generative Models** . You can find the poster presented to the [MICCAI Workshop](https://conferences.miccai.org/2024/en/workshops.asp) conference under [*Poster_MLMI24_60.pdf*](https://github.com/TransRad/USArt/blob/main/Poster_MLMI24_60.pdf).

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
Georgia KANLI, Daniele PERLO, Selma BOUDISSA, Radovan JIRIK and Olivier KEUNEN.


## Citation 
If you find **USArt** useful in your research, please use the following for citation.

G. Kanli, D. Perlo, S. Boudissa, Radovan Jirik  and O. Keunen. "Simultaneous Image Quality Improvement and Artefacts Correction in Accelerated MRI", 2024 MICCAI Workshop on Deep Generative Models, Marrakech, Morocco, 2024 (will be updated once paper published in proceedings )



