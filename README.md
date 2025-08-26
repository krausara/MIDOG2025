## MIDOG 2025 Track 2 Binary Classification

This repository contains the full pipeline submitted to the MICCAI MIDOG 2025 Challenge - Track 2.

The code, model evaluations, and all files required to reproduce the results are provided here. 

### Data sets
We used the offical dataset from the MIDOG Challenge, which is publically available:

    Weiss, V., et al. A Dataset of Atypical Vs Normal Mitoses Classification for MIDOG - 2025. Zenodo, 17 July 2025, https://doi.org/10.5281/zenodo.16044804.

In addition, we used the Octopath dataset also publically available:

    Shen, Z., et al. Omg-octo Atypical: A Refinement of the Original Omg-octo Database to Incorporate Atypical Mitoses. Zenodo, 18 July 2025, https://doi.org/10.5281/zenodo.16107743.

### Train and test splits
Each dataset (MIDOG and OCTO) was individually split at the **patient level** into an 80/20 train/test split using [`pat_split.py`](./pat_split.py).

After splitting, the training sets from both datasets were combined into a single training set, and the test sets were similarly combined into a single test set.

This procedure ensures that:
- Patients are uniquely assigned to either train or test (no patient overlap).
- Both datasets are equally represented in train and test.
- Neither dataset dominates the distribution of the final train/test sets.

### Pipeline
For internal testing, the [`pipeline.py`](./pipeline.py) script was used.  
For the official MIDOG Challenge submission, training was performed without an internal test set using [`pipeline_no_test.py`](./pipeline_no_test.py).