# Implementation details
This module contains routines to load data from DICOM files and to save reconstructed images to DICOM or NiFTI formats.

- **utils.py**  contains utility routines, mostly to sort dicom in correct order (contrast, slice, ny, nx) and to calculate spatial orientation of the image.
- **read.py** contains data reading routines (for DICOM only at the moment). It is intended to load contrast weighted images for parametric mapping.
- **write.py** contains data writing routines (both in DICOM and NiFTI formats). At present moment, it relies on DICOM template obtained from a scanner-generated DICOM files (using read.py). It is intended to write parametric maps obtained from fitting of scanner-reconstruced contrast-weighted images. TODO: design header generation from scratch, to enable full reconstruction pipeline from scanner raw data (or from ISMRMRD-converted files).
