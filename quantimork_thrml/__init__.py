"""
quantimork_thrml — QuantiMORK (§7.4) wavelet-sparse PC on thermodynamic factor graphs
======================================================================================

Core modules:
    haar            — Haar DWT / IDWT utilities
    wavelet_linear  — WaveletLinear: Haar → per-level Linear → Inverse Haar
"""
from quantimork_thrml.haar import haar_dwt_1d, haar_idwt_1d
from quantimork_thrml.wavelet_linear import WaveletLinear
