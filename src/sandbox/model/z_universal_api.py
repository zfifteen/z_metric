# Further extended z_universal_api.py with Invariance Band Support

# Global Invariant C (immutable)
UNIVERSAL_C = 3e8  # Example invariant

import numpy as np  # For band computation (e.g., ranges)


def compute_z_band(A, B_values, regime='physical'):
    """
    Computes Z and its 'band' (e.g., min/max/derivative spread) across B rates,
    illustrating additional info from C's invariance.

    Parameters:
    - A (float/int): Frame-dependent quantity
    - B_values (list of float/int): List of rates to compute band over
    - regime (str): Regime type

    Returns:
    - dict: Z values, band min/max, average derivative (historical info)
    """
    if regime == 'physical':
        C = UNIVERSAL_C
    elif regime == 'discrete':
        C = 10  # Regime-specific invariant proxy
    else:
        raise ValueError("Unsupported regime.")

    if C == 0:
        raise ValueError("C cannot be zero.")

    Z_values = [A * (B / C) for B in B_values]
    band_min = min(Z_values)
    band_max = max(Z_values)
    avg_dZ_dB = np.mean([A / C for _ in B_values])  # Average derivative as info band proxy

    return {
        'Z_values': Z_values,
        'band_range': (band_min, band_max),
        'avg_derivative': avg_dZ_dB  # Encodes historical/additional info
    }