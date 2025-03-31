"""
Class to encrypt and decrypt using simple censor shift using
simple poly alphabetic substitution
Name: MNA Ahimbisibwe
SN: 217005435
Model: UJ Mobile Phone Firmware Hacking Tool
"""
import string


class Encryption:
    # Initialize the class with a list of shifts
    def __init__(self, shifts):
        # Define the standard English alphabet for plaintext
        alpha = list(string.ascii_uppercase)
        self.plaintext_alphabet = ''.join(alpha)  # results "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.cipher_alphabets = []
        self.row_numbers = []

        for shift in shifts:
            substitution_alphabet, row_number = self.generate_substitution_alphabet(shift)
            self.cipher_alphabets.append(substitution_alphabet)
            self.row_numbers.append(row_number)

    def encrypt(self, word: str) -> str:
        word = word.upper()
        ciphertext = ""

        for i in range(len(word)):
            char = word[i]
            plaintext_index = self.plaintext_alphabet.find(char)

            if plaintext_index == -1:
                ciphertext += char
                continue

            substitution_alphabet = self.cipher_alphabets[i % len(self.cipher_alphabets)]
            substituted_char = substitution_alphabet[plaintext_index]
            ciphertext += substituted_char

        return ciphertext

    # Define a method to decrypt ciphertext into plaintext
    def decrypt(self, ciphertext: str) -> str:
        plaintext = ""

        for i in range(len(ciphertext)):
            char = ciphertext[i]
            # Find its index in the cipher alphabet
            cipher_index = self.cipher_alphabets[i % len(self.cipher_alphabets)].find(char)

            # If the character isn't in the cipher alphabet, just add it to the plaintext as is
            if cipher_index == -1:
                plaintext += char
                continue

            # Find the corresponding character in the plaintext alphabet
            original_char = self.plaintext_alphabet[cipher_index]
            # Add the original character to the plaintext
            plaintext += original_char

        return plaintext

    # Define a method to generate a cipher alphabet and row number from a shift
    def generate_substitution_alphabet(self, shift):
        shifted_alphabet = self.plaintext_alphabet[shift:] + self.plaintext_alphabet[:shift]
        row_number = shift + 1
        return shifted_alphabet, row_number

# # Define the list of shifts
# shifts = [4, 3, 1, 6, 7]
# # Create an instance of the Encryption class with these shifts
# encryption = Encryption(shifts)
# # Define the plaintext to be encrypted
# word = "JOHANNESBURG"
# # Encrypt the plaintext into ciphertext
# encrypted_word = encryption.encrypt(word)
# print(f"Plaintext: {word}")
# print(f"Ciphertext: {encrypted_word}")
#
# # Decrypt the ciphertext back into plaintext
# decrypted_word = encryption.decrypt(encrypted_word)
# print(f"Decrypted: {decrypted_word}")
