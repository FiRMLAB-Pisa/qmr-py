# Implementation details
This module contains routines to perform parameter inference from scanner-reconstructed contrast-weighted images.

- **utils.py**  contains utility routines to pre- / post- process contrast-weighted images / parametric maps (currently only binary masking of contrast-weighted images).
- **inversion_recovery.py** fitting routines to perform T1 mapping from Inversion Recovery Spin-Echo data.
- **multiecho.py** fitting routines to perform T2 / T2* mapping from Multiecho Spin-Echo / Gradient Echo data.
- **field_mapping.py** fitting routines to perform B1+ and B0 mapping (currently only Double-Angle method B1+ mapping).
