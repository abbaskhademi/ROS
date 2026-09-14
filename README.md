# Robust Optimization Under Sparse Uncertainty

This folder contains the experiment code and the input data used in the paper. Running the scripts creates new result files; no compiled paper, stored results, generated tables, or verification reports are included here.

## Code

- `code/SparseAGP.jl` — adaptive gradient projection, iterative hard thresholding, and sparse-optimization utilities.
- `code/src/SparseRobustRevision.jl` — shared Julia module used by the experiment drivers.
- `code/run_sparse_signal_recovery.jl` — sparse signal-recovery experiments.
- `code/run_sparse_logistic_regression.jl` — sparse logistic-regression experiments.
- `code/run_portfolio_public_data_scaled.jl` — robust portfolio experiment using the Fama–French data.
- `code/run_implementation_error_coupled.jl` — implementation-error experiments using the NETLIB instances.
- `code/Project.toml` and `code/Manifest.toml` — Julia environment and locked package versions for the main experiments.
- `code/wireless/DichasusWirelessData.jl` — loads and prepares the DICHASUS channel measurements.
- `code/wireless/WirelessSparseRO_v11.jl` — static robust wireless model and separation methods.
- `code/wireless/WirelessSparseARO_v11.jl` — adjustable robust wireless model and recovery methods.
- `code/wireless/run_wireless_paper_experiments_v11.jl` — driver for the static and adjustable wireless experiments.
- `code/wireless/extract_dichasus_snr.py` — converts the original DICHASUS TFRecord into the processed NPZ format.
- `code/wireless/requirements.txt` — Python packages needed only for that conversion.
- `code/wireless/Project.toml` and `code/wireless/Manifest.toml` — Julia environment and locked package versions for the wireless experiments.

## Data

- `data/fama_french/49_Industry_Portfolios.csv` — monthly returns for the 49 Fama–French industry portfolios.
- `data/libsvm/a5a`, `a6a`, `a7a`, and `a8a` — LIBSVM binary-classification datasets used for sparse logistic regression.
- `data/netlib_large/` — the SCFXM1, SCFXM3, 25FV47, WOODW, and FIT2D NETLIB linear-programming instances.
- `data/wireless_source/dichasus-d036-32subbands.npz` — processed DICHASUS measurements read directly by the Julia wireless experiments.

The original `dichasus-d036.tfrecords` download is not included because it is approximately 196 MB and exceeds GitHub's normal per-file limit. It can be downloaded from the public DICHASUS `dichasus-dxxx-reduced` collection and converted with `extract_dichasus_snr.py`; the included NPZ file is sufficient to run the experiments.

## Environments

Instantiate the main Julia environment with `julia --project=code -e 'using Pkg; Pkg.instantiate()'` and the wireless environment with `julia --project=code/wireless -e 'using Pkg; Pkg.instantiate()'`. The optimization experiments require MOSEK and a valid local MOSEK licence.
