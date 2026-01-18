import numpy as np
from src.lwe import LWE, center_modulo


def encrypt(lwe: LWE, message_bit: int) -> tuple:
    S = np.random.choice(lwe.m, size=lwe.m // 2, replace=False)
    a = np.sum(lwe.A[S, :], axis=0) % lwe.q
    c = (np.sum(lwe.b[S]) + message_bit * (lwe.q // 2)) % lwe.q
    return (a, c)


def decrypt(lwe: LWE, ciphertext: tuple[np.ndarray, np.ndarray]) -> int:
    a, c = ciphertext
    diff = center_modulo(c - a @ lwe._s, lwe.q)
    return 1 if abs(diff) > lwe.q // 4 else 0
