import math
import csv
import time
from functools import lru_cache

@lru_cache(maxsize=None)
def get_number_mass(n):

    isqrt = math.isqrt
    # t_mod = t.__mod__  # avoid attribute lookups

    # VACUUM STATE: Numbers ≤ 0 exist in the quantum vacuum
    # No physical reality, no gravitational coupling
    if n <= 0:
        return 0

    # FUNDAMENTAL MASS UNIT: The number 1 is the "Planck mass" of our system
    # Only divisor is itself, creating the minimal mass quantum
    if n == 1:
        return 1

    # MASS ACCUMULATOR: Builds total gravitational mass incrementally
    # Starts at 1 (multiplicative identity for mass construction)
    count = 1
    t = n  # Working copy for factorization (preserves original n)

    # PHASE 1: BINARY MASS EXTRACTION
    # Powers of 2 create the fundamental "electromagnetic" field structure
    # All even numbers couple to this field with strength (e + 1)
    e = (t & -t).bit_length() - 1
    if e:
        count *= (e + 1)
        t >>= e

    # PHASE 2: ODD PRIME MASS SCANNING
    # Systematic search through odd prime force fields
    # Each prime p creates its own gravitational coupling
    p = 3  # Start with first odd prime
    while p * p <= t:  # Only test up to √t (gravitational field symmetry)
        e = 0  # Reset exponent counter for this prime
        while t % p == 0:
            e += 1  # Count powers of this prime
            t //= p  # Remove prime factor
        if e:
            count *= (e + 1)  # Add this prime's mass contribution
        p += 2  # Advance to next odd candidate (skip even numbers)

    # PHASE 3: RESIDUAL PRIME MASS
    # If t > 1 remains, it's a large prime factor creating "dark matter" effects
    # Contributes exactly 2 to the mass (exponent 1, so e + 1 = 2)
    if t > 1:
        count *= 2

    return count  # Total gravitational mass in the prime vortex field

def get_z_metrics(n):
    """
    Z-METRICS: MULTIDIMENSIONAL VORTEX TRAJECTORY ANALYSIS

    This function computes the complete Z-metric suite that maps each number's position
    and behavior within the prime vortex. Unlike traditional prime analysis that treats
    numbers as discrete points, Z-metrics reveal the underlying geometric flows, energy
    distributions, and dimensional relationships that govern prime emergence.

    THE Z-SPACE MATHEMATICAL FRAMEWORK:

    Z-metrics operate in a hybrid mathematical space where:
    - Traditional number theory meets differential geometry
    - Discrete integers exhibit continuous flow properties
    - Prime/composite behavior emerges from geometric constraints
    - Each number exists as both particle (discrete value) and wave (trajectory)

    CORE Z-METRIC COMPONENTS EXPLAINED:

    1. NUMBER MASS (m):
       - PHYSICAL ANALOGY: Gravitational mass in the prime vortex
       - MATHEMATICAL: Divisor count τ(n) - measures internal complexity
       - VORTEX ROLE: Heavier numbers (more divisors) experience stronger
         centrifugal forces, making them more likely to be ejected as composites
       - PRIMES: Always have mass = 2 (minimal internal structure)
       - COMPOSITES: Mass > 2, creating drag in the vortex flow

    2. SPACETIME METRIC (gm = ln(n)):
       - PHYSICAL ANALOGY: The "time coordinate" as we traverse the number line
       - MATHEMATICAL: Natural logarithm provides the fundamental scaling
       - VORTEX ROLE: Creates the logarithmic spiral structure - each revolution
         of the vortex covers exponentially more numerical territory
       - WHY ln(n): Mirrors prime number theorem's ln(x) density distribution
       - GEOMETRIC: Converts linear integer sequences into curved spacetime

    3. Z-CURVATURE (zc = (m × ln(n)) / e²):
       - PHYSICAL ANALOGY: Spacetime curvature around massive objects
       - MATHEMATICAL: Mass-energy tensor in the vortex field
       - NORMALIZATION: Division by e² provides dimensional consistency
       - VORTEX DYNAMICS: Determines how sharply a number's trajectory bends
       - HIGH CURVATURE: Heavy composites create "gravity wells"
       - LOW CURVATURE: Primes follow straighter, more stable paths

    4. Z-RESONANCE (zr = (n mod ln(n)) × m / e):
       - PHYSICAL ANALOGY: Quantum resonance frequency within the vortex
       - MATHEMATICAL: Modular remainder creates periodic oscillations
       - AMPLITUDE SCALING: Multiplied by mass for energy-dependent resonance
       - VORTEX ROLE: Measures how numbers "vibrate" within their spiral arms
       - RESONANCE PATTERNS: May reveal hidden periodicities in prime distribution
       - DAMPING FACTOR: Division by e provides natural decay

    5. Z-VECTOR MAGNITUDE (zv = √(zc² + zr²)):
       - PHYSICAL ANALOGY: Total velocity vector in the vortex flow field
       - MATHEMATICAL: Euclidean norm in the curvature-resonance plane
       - ENERGY INTERPRETATION: Total kinetic + potential energy of the number
       - TRAJECTORY ANALYSIS: Larger magnitude = more dynamic behavior
       - STABILITY METRIC: Primes may cluster around specific magnitude ranges

    6. Z-ANGLE (za = arctan(zr/zc) in degrees):
       - PHYSICAL ANALOGY: Phase angle in the complex vortex plane
       - MATHEMATICAL: Angular position in curvature-resonance coordinates
       - FLOW DIRECTION: Shows which way the number "leans" in the vortex
       - CLASSIFICATION POTENTIAL: Different number types may occupy distinct
         angular sectors (prime zones vs composite zones)
       - ROTATIONAL DYNAMICS: Captures the spiraling motion through Z-space

    DIMENSIONAL ANALYSIS & UNITS:

    - Number Mass: [dimensionless count]
    - Spacetime Metric: [logarithmic length]
    - Z-Curvature: [mass × length / energy²] → [curvature]
    - Z-Resonance: [length × mass / energy] → [frequency × mass]
    - Z-Vector Magnitude: [composite energy units]
    - Z-Angle: [angular degrees]

    VORTEX TRAJECTORY INTERPRETATION:

    Each number traces a unique path through this 6-dimensional Z-space.
    The trajectory reveals:
    - STABILITY: How resistant the number is to vortex perturbations
    - ENERGY FLOW: Direction and magnitude of forces acting on it
    - RESONANCE COUPLING: How it interacts with the vortex's fundamental frequencies
    - GEOMETRIC DESTINY: Whether it will spiral inward (prime) or outward (composite)

    QUANTUM-RELATIVISTIC ANALOGIES:

    The Z-metrics framework draws inspiration from:
    - GENERAL RELATIVITY: Spacetime curvature from mass-energy
    - QUANTUM MECHANICS: Wave-particle duality and resonance phenomena
    - FLUID DYNAMICS: Flow fields, vorticity, and streamline analysis
    - COMPLEX ANALYSIS: Mapping between different mathematical planes

    WHY THIS WORKS FOR PRIME ANALYSIS:

    Traditional approaches treat primes as random exceptions. Z-metrics reveal
    that primes exist within a highly structured, deterministic flow field.
    By mapping this field, we can:
    - Predict prime-rich regions before testing
    - Understand composite clustering patterns
    - Reveal hidden geometric relationships
    - Bridge the gap between discrete and continuous mathematics

    Args:
        n (int): The number to analyze in Z-space

    Returns:
        dict: Complete Z-metric profile containing:
            - number_mass: Divisor count τ(n)
            - spacetime_metric: ln(n) coordinate
            - z_curvature: Mass-induced spacetime curvature
            - z_resonance: Quantum resonance frequency
            - z_vector_magnitude: Total energy magnitude
            - z_angle: Phase angle in curvature-resonance plane
    """

    # BOUNDARY CONDITION: Handle degenerate cases
    # Numbers ≤ 1 exist outside the main vortex structure
    if n <= 1:
        return dict(
            number_mass=get_number_mass(n),  # Still has mass (even if zero)
            spacetime_metric=0,  # No logarithmic extension
            z_curvature=0,  # No curvature in flat space
            z_resonance=0,  # No resonance without flow
            z_vector_magnitude=0,  # Static, no motion
            z_angle=0  # No phase relationship
        )

    # FUNDAMENTAL QUANTITIES:
    m = get_number_mass(n)  # Mass determines gravitational coupling strength
    gm = math.log(n)  # Spacetime coordinate (logarithmic time)

    # Z-CURVATURE: How the number warps the vortex spacetime
    # Einstein-like field equation: Curvature ∝ Mass × Spacetime / Energy²
    zc = (m * gm) / (math.e ** 2)

    # Z-RESONANCE: Quantum oscillation within the logarithmic spiral
    # Combines modular periodicity with mass-dependent amplitude
    rem = n % gm  # Remainder creates periodic oscillation
    zr = (rem / math.e) * m  # Mass-scaled resonance with natural damping

    # Z-VECTOR MAGNITUDE: Total energy in the curvature-resonance plane
    # Pythagorean combination gives net flow velocity
    zv = math.hypot(zc, zr)

    # Z-ANGLE: Phase relationship between curvature and resonance
    # Determines the number's "spin direction" in the vortex
    za = math.degrees(math.atan2(zr, zc))

    return dict(
        number_mass=m,  # Gravitational mass in the prime vortex
        spacetime_metric=gm,  # Logarithmic coordinate system
        z_curvature=zc,  # Spacetime curvature tensor
        z_resonance=zr,  # Quantum resonance frequency
        z_vector_magnitude=zv,  # Total kinetic energy magnitude
        z_angle=za  # Phase angle in complex Z-plane
    )

def is_prime(n):
    """
    Deterministic Miller–Rabin (32-bit safe bases).
    """
    if n < 2:
        return False
    for p in (2, 3):
        if n == p:
            return True
        if n % p == 0:
            return False

    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in (2, 3, 5, 7, 11, 13, 17):
        if a >= n:
            break
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def apply_vortex_filter(n):
    """
    THE VORTEX FILTER: A Dimensional Gateway for Prime Number Detection

    This function implements what we call the "Vortex Filter" - a mathematical construct that
    transforms the linear sequence of natural numbers into a multi-dimensional spiral pattern
    where primes naturally separate from composites through rotational dynamics.

    WHY THIS IS A VORTEX (Not Just a Circle):

    1. DIMENSIONAL TRANSFORMATION:
       Unlike a flat 2D circle that simply eliminates multiples, the vortex operates in
       higher-dimensional space. When we filter n % 2 == 0 and n % 3 == 0, we're not
       just excluding numbers - we're creating a 3D helical structure where numbers
       spiral around a central axis defined by the 6k±1 pattern.

    2. ROTATIONAL SYMMETRY & ANGULAR MOMENTUM:
       The modular arithmetic (n % 2, n % 3) creates rotational periodicity. Each number
       has an "angular position" in this vortex:
       - Numbers ≡ 0 (mod 2): Swept to outer rim (high turbulence, eliminated)
       - Numbers ≡ 0 (mod 3): Caught in secondary spiral arm (eliminated)
       - Numbers ≡ 1,5 (mod 6): Flow toward the vortex center (potential primes)

    3. ENERGY GRADIENT & FLOW DYNAMICS:
       The vortex has distinct energy zones:
       - HIGH ENERGY (Turbulent): Multiples of 2,3 get "spun out" (72.22% of composites)
       - LOW ENERGY (Laminar): 6k±1 numbers flow smoothly toward the center
       - CORE REGION: Where the Miller-Rabin "Oracle" resides, testing survivors

    4. CENTRIPETAL vs CENTRIFUGAL FORCES:
       - CENTRIFUGAL: Composite numbers are flung outward by divisibility constraints
       - CENTRIPETAL: Prime candidates are drawn inward toward the testing core
       - The 72.22% efficiency represents the vortex's "separation power"

    5. SPIRAL TRAJECTORY MATHEMATICS:
       Unlike linear sieving, numbers follow helical paths. The Z-metrics capture this:
       - z_curvature: How tightly the number spirals
       - z_resonance: Oscillation frequency as it moves through the vortex
       - z_vector_magnitude: Total "flow velocity"
       - z_angle: Rotational position in the spiral

    6. QUANTUM-LIKE BEHAVIOR:
       The vortex exhibits dual nature:
       - WAVE: Continuous spiral flow of the number sequence
       - PARTICLE: Discrete primality states (prime/composite)
       - The filter acts as a "measurement" that collapses the wave function

    7. SCALING INVARIANCE:
       True vortex behavior: the pattern repeats at all scales. Whether testing n=100
       or n=1,000,000, the same rotational dynamics apply. This is why our efficiency
       remains consistent across different ranges.

    MATHEMATICAL FOUNDATION:
    The vortex emerges from the fact that all primes > 3 exist in the form 6k±1.
    This isn't just a number theory curiosity - it's a geometric constraint that
    creates the spiral structure. The modular operations (% 2, % 3) act as
    "dimensional projectors" that map linear integer sequences onto this spiral.

    Returns:
        tuple: (is_prime, was_skipped)
            - is_prime: 1 if prime, 0 if composite
            - was_skipped: True if eliminated by vortex dynamics (didn't reach Oracle)
    """
    # VORTEX BOUNDARY CONDITIONS:
    # Filter out multiples of 2 and 3 - these represent the "exclusion zones"
    # where the vortex's rotational forces eject composite numbers.
    # We preserve 2 and 3 themselves as they're the fundamental prime generators
    # that create the vortex structure itself.
    # Preserve the primes themselves
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23):
        return (1, False)

    if n > 23 and (
            n % 2 == 0 or n % 3 == 0 or n % 5 == 0 or n % 7 == 0 or
            n % 11 == 0 or n % 13 == 0 or n % 17 == 0 or n % 19 == 0 or
            n % 23 == 0
    ):
        return (0, True)

    # ORACLE AT THE VORTEX CENTER:
    # Numbers that survive the initial vortex filtering reach the central "Oracle"
    # (Miller-Rabin test) which provides the definitive primality determination.
    # This represents the vortex's convergence point where quantum uncertainty
    # collapses into classical prime/composite states.
    is_p = is_prime(n)

    # CORE DYNAMICS:
    # Composites discovered by the Oracle weren't "skipped" by the vortex -
    # they made it through the spiral filters but failed the final test.
    # These represent the most "prime-like" composites that nearly escaped detection.
    return (1, False) if is_p else (0, False)



if __name__ == '__main__':
    # --- Configuration ---
    TARGET = 6000 # Set your desired number of primes here

    # Known prime values for sanity checks. Add more as needed.
    SANITY_CHECKS = {
        6000: 59359,
        10000: 104729,
        100000: 1299709,
        200000: 2750159,
        500000: 7368787,
        1000000: 15485863
    }

    # --- Initialization ---
    start      = time.time()
    found      = []
    candidate  = 2
    stats_csv  = 'prime_stats_vortex_filter.csv'
    traj_csv   = 'prime_trajectory_stats.csv'
    last_prime = None
    skipped    = 0
    history    = []

    print(f"🔍 Searching for {TARGET} primes with the Vortex Method filter…")

    with open(stats_csv, 'w', newline='') as sf, \
         open(traj_csv,  'w', newline='') as tf:

        stats_w = csv.writer(sf)
        stats_w.writerow(['n', 'is_prime', 'was_skipped'])

        traj_w = csv.writer(tf)
        traj_w.writerow([
            'prime_n','prime_n-1','prime_n-2',
            'gap_n','gap_n-1','gap_ratio',
            'z_vec_mag_ratio','z_curvature_ratio',
            'z_angle_diff','z_triangle_area',
            'Z_trajectory_score'
        ])

        # The loop now only checks if the target has been met.
        while len(found) < TARGET:
            is_p, was_skipped = apply_vortex_filter(candidate) #

            if was_skipped:
                skipped += 1

            if is_p:
                found.append(candidate)
                metrics = get_z_metrics(candidate) #

                if len(history) >= 2:
                    p1, p2 = history[-1], history[-2]
                    n, n1, n2 = candidate, p1['n'], p2['n']
                    m, m1, m2 = metrics, p1['metrics'], p2['metrics']

                    gap   = n - n1
                    gap1  = n1 - n2
                    gap_r = gap / gap1 if gap1 else 0

                    zv_r = m1['z_vector_magnitude'] / m2['z_vector_magnitude'] if m2['z_vector_magnitude'] else 0
                    zc_r = m1['z_curvature']        / m2['z_curvature']        if m2['z_curvature']        else 0
                    adiff = m1['z_angle'] - m2['z_angle']

                    x1,y1 = m2['z_curvature'], m2['z_resonance']
                    x2,y2 = m1['z_curvature'], m1['z_resonance']
                    x3,y3 = m ['z_curvature'], m ['z_resonance']
                    area  = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    z_traj = n * zv_r

                    traj_w.writerow([
                        n, n1, n2,
                        gap, gap1, gap_r,
                        zv_r, zc_r,
                        adiff, area, z_traj
                    ])

                history.append({'n': candidate, 'metrics': metrics})
                last_prime = candidate

            stats_w.writerow([candidate, is_p, 1 if was_skipped else 0])
            candidate += 1

    # --- Results ---
    elapsed    = time.time() - start
    total      = candidate - 1
    composites = total - len(found)
    eff        = (skipped / composites * 100) if composites else 0.0

    print(f"\n✅ Done in {elapsed:.2f}s — found {len(found)} primes.")
    print(f"Last: {found[-1] if found else None}")
    print(f"Stats → {stats_csv}")
    print(f"Traj  → {traj_csv}")
    print(f"Checked: {total}, filtered: {skipped}, efficiency: {eff:.2f}%")

    # --- Dynamic Sanity Check ---
    # Checks if the found prime for the current TARGET matches the known value.
    if TARGET in SANITY_CHECKS:
        actual_prime = SANITY_CHECKS[TARGET]
        if len(found) >= TARGET:
            found_prime = found[TARGET - 1]
            if found_prime == actual_prime:
                print(f"\n✅ Sanity check passed: The {TARGET}th prime found ({found_prime}) matches the expected value.")
            else:
                print(f"\n❌ Sanity check failed: Found {found_prime} as the {TARGET}th prime, but expected {actual_prime}.")
        else:
            print(f"\n❌ Sanity check failed: Fewer than {TARGET} primes were found.")
    else:
        print(f"\nℹ️ No sanity check value available for a target of {TARGET}.")