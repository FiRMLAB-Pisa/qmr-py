# Implementation details
This module contains routines to perform parameter inference from scanner-reconstructed contrast-weighted images.

- **inversion_recovery.py** fitting routines to perform T1 mapping from Inversion Recovery Spin-Echo data.
- **mp2rage.py** fitting routines to perform T1 mapping from MP2RAGE (and similar approaches, such as FLAWS) data.
- **multiecho.py** fitting routines to perform T2 / T2* mapping from Multiecho Spin-Echo / Gradient Echo data.
- **field_mapping.py** fitting routines to perform B1+ and B0 mapping (currently only Double-Angle method B1+ mapping and Dual Echo B0 mapping).
- **helmholtz_ept** fitting routines to perform conductivity mapping from the phase of bSSFP, Spin-Echo or UTE data (currently, laplacian-based only).
- **water_ept** fitting routines to perform conductivity and permittivity mapping from input quantitative T1 data.
- **utils.py**  contains utility routines to pre- / post- process contrast-weighted images / parametric maps (currently only magnitude-based binary masking of contrast-weighted images).
