# MISR CMV and ERA5 Analysis Pipeline

> *Code archive for Larry Di Girolamo1, Guangyu Zhao, Gan Zhang, Zhuo Wang, Jesse Loveridge, and Arka Mitra, "Decadal changes in atmospheric circulation detected in cloud motion vectors " for Nature MS  2024-10-22368A-Z
*  
> DOI: _placeholder_
> 
## Overview
This repository contains the Python scripts used to  
1. Aggerate MISR CMV and ERA5 data onto global regular 1-degree grid,  
2. compute MISR CMV  statistics and trends,  
3. generate the main text figures.

## Dependencies
- Python ≥ 3.10  
- `numpy`, `xarray`, `scipy`, `pandas`, `matplotlib`, `cartopy`  
- Optional HPC tools: `dask`, `mpi4py`

Create an environment:

```bash
conda env create -f environment.yml
conda activate mmcth
