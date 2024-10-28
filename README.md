# Optimal Weighted Random Forests

This repository contains the data sets and code for the paper "Optimal Weighted Random Forests," accepted by the Journal of Machine Learning Research (Volume 25, 2024, pp. 1-81).

## Repository Structure

- **dataset/**: This folder contains the 12 real data sets used in this study (see Table 2 in the paper), along with their sources.
  
- **owrf.R**: This file provides the code for our proposed algorithms, $1step-WRF_{\mathrm{opt}}$ and $2steps-WRF_{\mathrm{opt}}$, as well as implementations of competitor methods, using CART trees, in accordance with Section 4.1.1 of the paper.

- **owrf_SUT.R**: This file contains the code for our proposed algorithms, \( 1\text{step-}WRF_{\text{opt}} \) and \( 2\text{steps-}WRF_{\text{opt}} \), using SUT trees, following Section 4.1.2 of the paper.

- **owrf_robust.R**: This file includes code for testing the robustness of the random-forest hyperparameters "ntree" and "nodesize," as demonstrated in Figures D.12 - D.23 in Appendix D.

- **owrf_withNoise.R**: This file provides code for the proposed algorithms on semi-synthetic data sets with injected noise, according to Section 4.2.1 of the paper.

- **featureEngineering.py**: This script creates semi-synthetic data sets, based on real data sets, for Section 4.2.2 of the paper.

## Instructions

The codes are configured to use the "housing" data set as an example. To apply the codes to other data sets, replace "housing" with the desired data set in the data input area.

---

Please refer to the paper for more details on the methodology and experimental setup.
