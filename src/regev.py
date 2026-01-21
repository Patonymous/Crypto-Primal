import numpy as np
from src.lwe import LWE, center_modulo

# Implementacja oparta na:
# https://link.springer.com/chapter/10.1007/978-981-19-7644-5_4
# https://en.wikipedia.org/wiki/Learning_with_errors#Public-key_cryptosystem


def encrypt(lwe: LWE, message: str) -> list[tuple[np.ndarray, int]]:
    bytes = message.encode()
    bits = []
    for byte in bytes:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return [encrypt_bit(lwe, bit) for bit in bits]


def encrypt_bit(lwe: LWE, message_bit: int) -> tuple[np.ndarray, int]:
    S = np.random.choice(lwe.m, size=lwe.m // 2, replace=False)
    c = (np.sum(lwe.b[S]) + message_bit * (lwe.q // 2)) % lwe.q
    return (S, c)


def decrypt(lwe: LWE, ciphertexts: list[tuple[np.ndarray, int]]) -> str:
    bits = [decrypt_bit(lwe, ciphertext) for ciphertext in ciphertexts]
    bytes_arr = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte |= bits[i + j] << (7 - j)
        bytes_arr.append(byte)
    try:
        return bytes_arr.decode('utf-8')
    except UnicodeDecodeError:
        # Jeśli nie można zdekodować jako UTF-8, zwróć reprezentację hex
        return f"<błąd dekodowania: {bytes_arr.hex()}>"


def decrypt_bit(lwe: LWE, ciphertext: tuple[np.ndarray, int]) -> int:
    S, c = ciphertext
    a = np.sum(lwe.A[S, :], axis=0) % lwe.q
    diff = center_modulo(c - a @ lwe._s, lwe.q)
    return 1 if abs(diff) > lwe.q // 4 else 0
