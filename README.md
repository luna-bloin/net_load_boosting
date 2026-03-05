# Git repository for the paper "Climate change impacts on net load under technological uncertainty in European power systems"

This repository contains the code necessary to reproduce the results found in Bloin-Wibe et al., 2026 (1). It calculates Europe-wide net load (residual demand after generation) for different energy scenarios and climate periods: 
<img width="835" height="570" alt="image" src="https://github.com/user-attachments/assets/2b5cb59d-d0ca-4863-a166-b37568b192aa" />

To use this code, we assume that energy variables for the different climate periods are processed through Climate2Energy (2, github repository [here](https://github.com/jwohland/Climate2Energy)). 

The repository is organized as follows: 
- `code`
  * `plots_cc_impacts` contains the pythoin scripts to reproduce the manuscript figures
  * `postprocess` contains files for postprocessing large files (internally needed for storage reasons)
  * `preprocess` contains all files necessary to go from energy variables per country to European-level net load
  * `sensitivity_analyses` contains jupyter notebooks that calculates all sensitivity analyses that were performed for robustness
  * `utils` contains all necessary helper functions
- `inputs` includes all necessary files for net load conversion (installed capacity assumptions, average hydropower storage levels...)


(1) Bloin-Wibe, L. et al. Climate change impacts on net load under technological uncertainty in European power systems. Preprint at https://doi.org/10.48550/arXiv.2512.13461 (2025).

(2) Wohland, J. et al. Climate2Energy: a framework to consistently include climate change into energy system modeling. Environ. Res.: Energy 2, 041001 (2025).

