from sympy import primerange, primefactors
from math import gcd, lcm

def char_prime_map(charset):
    primes = list(primerange(2, 10000))
    return {char: primes[i] for i, char in enumerate(sorted(set(charset)))}

def encode_string(s, mapping):
    return prod(mapping[c] for c in set(s) if c in mapping)

def decode_number(n, reverse_map):
    factors = primefactors(n)
    return ''.join(sorted(reverse_map[p] for p in factors))

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

if __name__ == "__main__":
    # Input poetic lines
    line1 = "i took the one less traveled by"
    line2 = "and that has made all the difference"

    # Normalize to characters (lowercase, no spaces)
    chars1 = line1.replace(" ", "").lower()
    chars2 = line2.replace(" ", "").lower()
    charset = set(chars1 + chars2)

    # Prime mapping
    mapping = char_prime_map(charset)
    reverse_map = {v: k for k, v in mapping.items()}

    # Encode lines
    n1 = encode_string(chars1, mapping)
    n2 = encode_string(chars2, mapping)

    print("\n📝 Encoded Meaning from Poetry\n")
    print(f"Line 1: \"{line1}\"")
    print(f"→ Encoded as: {n1}")
    print(f"→ Decoded: \"{decode_number(n1, reverse_map)}\"")

    print(f"\nLine 2: \"{line2}\"")
    print(f"→ Encoded as: {n2}")
    print(f"→ Decoded: \"{decode_number(n2, reverse_map)}\"")

    print("\n🔍 Semantic Arithmetic (Logic via Numbers)\n")
    inter = gcd(n1, n2)
    union = lcm(n1, n2)
    diff = n1 // inter

    print(f"Shared semantic content (intersection):\n→ GCD: {inter}\n→ Characters: \"{decode_number(inter, reverse_map)}\"")

    print(f"\nCombined meaning (union):\n→ LCM: {union}\n→ Characters: \"{decode_number(union, reverse_map)}\"")

    print(f"\nUnique to Line 1 (difference):\n→ Diff: {diff}\n→ Characters: \"{decode_number(diff, reverse_map)}\"")
