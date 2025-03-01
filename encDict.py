import hashlib
import numpy as np
import pickle
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

aes_key = b'ThisIsA32ByteKeyForAES256Encryption!'[:32]

# def encrypt(data):
#     return encrypt_data_deterministic(data, aes_key)
#
# def encrypt_nd(data):
#     return encrypt_data_nondeterministic(data, aes_key)

def generate_deterministic_iv(data):
    return hashlib.sha256(data).digest()[:16]

def encrypt_data_deterministic(data, key=aes_key):
    serialized_data = pickle.dumps(data)
    iv = generate_deterministic_iv(serialized_data)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(serialized_data, AES.block_size))
    return iv + ciphertext

def encrypt_data_nondeterministic(data, key=aes_key):
    serialized_data = pickle.dumps(data)
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(serialized_data, AES.block_size))
    return iv + ciphertext

def decrypt(encrypted_data, key=aes_key):
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return pickle.loads(decrypted_data)

# def decrypt(encrypted_data, key=aes_key):
#     iv = encrypted_data[:16]
#     ciphertext = encrypted_data[16:]
#     cipher = AES.new(key, AES.MODE_CBC, iv)
#     decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)
#     return pickle.loads(decrypted_data)

def encrypt_dict(input_dict):
    encrypted_dict = {}
    for key, value in input_dict.items():
        key_str = str(key).encode()
        encrypted_key = encrypt_data_deterministic(key_str, aes_key)
        encrypted_value = encrypt_data_nondeterministic(value, aes_key)
        encrypted_dict[encrypted_key] = encrypted_value
    return encrypted_dict

def decrypt_dict(encrypted_dict):
    decrypted_dict = {}
    for encrypted_key, encrypted_value in encrypted_dict.items():
        decrypted_key_str = decrypt(encrypted_key, aes_key)
        decrypted_key = eval(decrypted_key_str.decode())
        decrypted_value = decrypt(encrypted_value, aes_key)
        decrypted_dict[decrypted_key] = decrypted_value
    return decrypted_dict

# input_dict = {
#     (1, 2): np.array([[1.2, 2.3], [3.4, 4.5]]),
#     (3, 4): np.array([[5.6, 6.7], [7.8, 8.9], [9.1, 10.2]])
# }
#
# encrypted_dict = encrypt_dict(input_dict)
# print("Encrypted Dictionary:")
# print(encrypted_dict)
#
# decrypted_dict = decrypt_dict(encrypted_dict)
# print("\nDecrypted Dictionary:")
# print(decrypted_dict)
