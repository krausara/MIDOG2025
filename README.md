## MIDOG 2025 Track 2 Binary Classification

This repository contains the full pipeline submitted to the MICCAI MIDOG 2025 Challenge - Track 2.

The code, model evaluations, and all files required to reproduce the results are provided here. 

### Data sets
We used the AmiBr dataset available at: 

    https://github.com/DeepMicroscopy/AMi-Br/tree/main/patches

    Bertram, C.A. et al. (2025). Histologic Dataset of Normal and Atypical Mitotic Figures on Human Breast Cancer (AMi-Br). In: Palm, C., et al. Bildverarbeitung für die Medizin 2025. BVM 2025. Informatik aktuell. Springer Vieweg, Wiesbaden.

We added the offical dataset from the MIDOG Challenge, without the overlap with Amibr. This dataset is publically available:

    Weiss, V., et al. A Dataset of Atypical Vs Normal Mitoses Classification for MIDOG - 2025. Zenodo, 17 July 2025, https://doi.org/10.5281/zenodo.16044804.

Lastly, we used the Octopath dataset also publically available:

    Shen, Z., et al. Omg-octo Atypical: A Refinement of the Original Omg-octo Database to Incorporate Atypical Mitoses. Zenodo, 18 July 2025, https://doi.org/10.5281/zenodo.16107743.


### Train and test split
All three datasets were merged into one cvs with the ['merge_datasets.py'](./merge_datasets.py). Here the duplicated data from Amibr got removed from Midog. 
The merged dataset was then split at **patient level** into an 80/20 train/test split using [`pat_split.py`](./pat_split.py).
The used split is provided in the pat_split folder.

### Pipeline
For internal testing, the [`pipeline.py`](./pipeline.py) script was used.  
For the official MIDOG Challenge submission, training was performed without an internal test set using [`pipeline_no_test.py`](./pipeline_no_test.py).