def is_prime_via_resonance(n, p_set=primes_up_to_1000):
    residues = [n % p for p in p_set]
    spectrum = np.fft.fft(residues)
    if has_peak_at(spectrum, f=1/7.0):  # Resonant frequency
        return True